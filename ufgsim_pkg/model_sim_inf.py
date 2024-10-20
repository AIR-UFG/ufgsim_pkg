from colorcloud.UFGsim2024infufg import SemanticSegmentationSimLDM, ProjectionSimTransform, ProjectionSimVizTransform

from colorcloud.biasutti2019riu import RIUNet
from colorcloud.biasutti2019riu import SemanticSegmentationTask as RIUTask
from colorcloud.chen2020mvlidarnet import MVLidarNet
from colorcloud.chen2020mvlidarnet import SemanticSegmentationTask as MVLidarTask

from colorcloud.behley2019iccv import SphericalProjection
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from ros2_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
import cv2
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import yaml
import os
import time
from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory

class SIMModelInferencer(Node):
	def __init__(self):
		super().__init__('model_sim')
		'''
		future parameters
		'''
		self.declare_parameter('model_name', 'RIUNet')
		self.declare_parameter('in_channels', 4)
		self.declare_parameter('n_classes', 13)
		self.declare_parameter('fov_up', 15.0)
		self.declare_parameter('fov_down', -15.0)
		self.declare_parameter('width', 440)
		self.declare_parameter('height', 16)
		self.declare_parameter('yaml_path', os.path.join(get_package_share_directory('ufgsim_pkg'),'config','ufg-sim.yaml'))
		self.declare_parameter('model_path', '/root/ros2_ws/src/ufgsim_pkg/config/ufgsim_riunet_torch.pt')

		model_name = self.get_parameter('model_name').get_parameter_value().string_value
		in_channels = self.get_parameter('in_channels').get_parameter_value().integer_value
		n_classes = self.get_parameter('n_classes').get_parameter_value().integer_value
		fov_up = self.get_parameter('fov_up').get_parameter_value().double_value
		fov_down = self.get_parameter('fov_down').get_parameter_value().double_value
		width = self.get_parameter('width').get_parameter_value().integer_value
		height = self.get_parameter('height').get_parameter_value().integer_value
		yaml_path = self.get_parameter('yaml_path').get_parameter_value().string_value
		self.model_path = self.get_parameter('model_path').get_parameter_value().string_value

		self.spherical_projection = SphericalProjection(fov_up, fov_down, width, height)

		self.projection_transform = ProjectionSimTransform(self.spherical_projection)


		self.logger = get_logger("inference_callback")

		self.subscription = self.create_subscription(
			PointCloud2,
			'/velodyne_points',
			self.inference_callback,
			100
		)
		self.publisher_stage1 = self.create_publisher(Image, '/segmentation_model', 10)
		self.publisher_acc = self.create_publisher(Image, '/segmentation_groundtruth', 10)
		self.publisher_stage2 = self.create_publisher(PointCloud2, '/segmentation_pc2_model', 10)
		self.bridge = CvBridge()
		self.model, self.task_class = self.load_model(model_name, in_channels, n_classes)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.loaded_model = torch.load(self.model_path).to(self.device)
		self.loaded_model.eval()
		self.loss = CrossEntropyLoss(reduction="none")
		with open(yaml_path, 'r') as file:
			metadata = yaml.safe_load(file)

		self.learning_map = metadata['learning_map']
		max_key = sorted(self.learning_map.keys())[-1]
		self.learning_map_np = np.zeros((max_key+1,), dtype=int)
		for k, v in self.learning_map.items():
			self.learning_map_np[k] = v

		self.learning_map_inv = metadata['learning_map_inv']
		self.learning_map_inv_np = np.zeros((len(self.learning_map_inv),))
		for k, v in self.learning_map_inv.items():
			self.learning_map_inv_np[k] = v

		self.color_map_bgr = metadata['color_map']
		max_key = sorted(self.color_map_bgr.keys())[-1]
		self.color_map_rgb_np = np.zeros((max_key+1,3))
		for k,v in self.color_map_bgr.items():
			self.color_map_rgb_np[k] = np.array(v[::-1], np.float32)

		self.projectionviz_transform = ProjectionSimVizTransform(self.color_map_rgb_np, self.learning_map_inv_np)
		
		# model warmup runs
		input_data = torch.randn(1,4,16,440).to(self.device)
		for i in range(5):
			with torch.no_grad():
				_ = self.loaded_model(input_data)

	def load_model(self, model_name, in_channels, n_classes):
		"""
		Dynamically load the model class based on the model_name parameter.
		"""
		if model_name == 'RIUNet':
			model = RIUNet(in_channels=in_channels, hidden_channels=(64, 128, 256, 512), n_classes=n_classes)
			task_class = RIUTask
		elif model_name == 'MVLidarNet':
			print("mvlidarnet")
			model =  MVLidarNet(in_channels=in_channels, n_classes=n_classes)
			task_class = None
		else:
			raise ValueError(f"Model {model_name} not supported")

		return model, task_class
	
	
	def inference_callback(self, msg):
		
		#start_time = time.time()
		pointcloud = structured_to_unstructured(pointcloud2_to_array(msg))
		
		pointcloud_xyz = pointcloud[:,:3]

		item = {
			'frame': pointcloud_xyz,
			'label': None,
			'mask': None
		}
		
		item = self.projection_transform(item)

		# Convert to float32 before passing to the model
		frame = item['frame'].astype(np.float32)
				
		with torch.no_grad():
			frame = item['frame']
			frame = np.expand_dims(frame, 0)
			frame = torch.from_numpy(frame).to(self.device)
			frame = frame.permute(0, 3, 1, 2).float() # (N, H, W, C) -> (N, C, H, W)
			y_hat = self.loaded_model(frame).squeeze()
			argmax = torch.argmax(y_hat, dim=0)
			pred = argmax.to(torch.uint8).cpu().numpy()
		
		item['label'] = pred
		
		# prediction imgmsg
		colored_pred = self.projectionviz_transform(item)
		colored_pred_uint8 = colored_pred['label'].astype(np.uint8)
		inferred_frame_msg = self.bridge.cv2_to_imgmsg(colored_pred_uint8, encoding='passthrough')
		self.publisher_stage1.publish(inferred_frame_msg)
		
		# groundtruth accuracy imgmsg
		
		label = pointcloud[:, 3].astype(np.uint32)
		label = self.learning_map_np[label]
		mask = label != 0
		
		gt_item = {
			'frame': pointcloud_xyz,
			'label': label,
			'mask': mask
		}
		gt_item = self.projection_transform(gt_item)
		gt_item = self.projectionviz_transform(gt_item)
		gt_item_uint8 = gt_item['label'].astype(np.uint8)

		disparity_frame_msg = self.bridge.cv2_to_imgmsg(gt_item_uint8, encoding='passthrough')
		self.publisher_acc.publish(disparity_frame_msg)
		
		# online loss calculation per inference published
		"""
		mask_for_loss = gt_item['mask']
		mask_for_loss = np.expand_dims(mask_for_loss,0)
		label_for_loss = gt_item['label']
		label_for_loss[~gt_item['mask']] = 0
		label_for_loss = np.expand_dims(label_for_loss,0)
		y_hat_for_loss = np.array(y_hat.cpu())
		y_hat_for_loss = np.expand_dims(y_hat_for_loss,0)
		# #self.logger.info(f"label shape: {label_for_loss.shape} - y_hat shape: {y_hat.shape}")
		loss = self.loss(torch.from_numpy(y_hat_for_loss).to(self.device), torch.from_numpy(label_for_loss).to(self.device))
		loss = loss[mask_for_loss]
		loss = loss.mean()
		self.logger.info(f"loss: {loss} device: {self.device}")
		"""

		# stage 2		
		frame = item['frame']
		frame = np.transpose(frame, (2,0,1))
		frame = frame[:3,:,:]
		pred_expanded = np.expand_dims(pred, axis=0)
		frame_with_pred = np.concatenate((frame, pred_expanded))
		frame_with_pred = np.transpose(frame_with_pred, (1,2,0))
		#frame_with_pred = frame_with_pred.reshape(-1)
		
		# Need to structure the array for array_to_pointcloud2
		dtype = np.dtype([('stage2dtype1', 'i1'), ('stage2dtype2', 'i1'), ('stage2dtype3', 'i1'), ('stage2dtype4', 'i1')])
		data_reshaped = frame_with_pred.reshape(-1, 4)
		frame_with_pred_structured = unstructured_to_structured(data_reshaped, dtype)
		frame_with_pred_structured = frame_with_pred_structured.reshape(16,440)
		inferred_point_cloud_msg = array_to_pointcloud2(frame_with_pred_structured, frame_id="frame")
		self.publisher_stage2.publish(inferred_point_cloud_msg)

		#end_time = time.time()
		#print(f"time taken {end_time - start_time:.6f} seconds")
def main(args=None):
	rclpy.init(args=args)
	model_inferencer = SIMModelInferencer()

	try:
		rclpy.spin(model_inferencer)
	except KeyboardInterrupt:
		pass
	finally:
		model_inferencer.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
    main()