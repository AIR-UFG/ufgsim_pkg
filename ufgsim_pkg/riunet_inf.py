from colorcloud.UFGsim2024infufg import SemanticSegmentationSimLDM, ProjectionSimTransform, ProjectionSimVizTransform
from colorcloud.biasutti2019riu import SemanticSegmentationTask, RIUNet
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
from ament_index_python.packages import get_package_share_directory

class RIUNetInferencer(Node):
	def __init__(self):
		super().__init__('riunet_sim')
		'''
		future parameters
		'''
		self.declare_parameter('fov_up', 15.0)
		self.declare_parameter('fov_down', -15.0)
		self.declare_parameter('width', 440)
		self.declare_parameter('height', 16)
		self.declare_parameter('yaml_path', '/home/air/ros2_ws/src/ufgsim_pkg/config/ufg-sim.yaml')
		self.declare_parameter('model_path', '/home/air/ros2_ws/src/ufgsim_pkg/ufgsim_pkg/ufgsim_riunet_4292.ckpt')
        
		fov_up = self.get_parameter('fov_up').get_parameter_value().double_value
		fov_down = self.get_parameter('fov_down').get_parameter_value().double_value
		width = self.get_parameter('width').get_parameter_value().integer_value
		height = self.get_parameter('height').get_parameter_value().integer_value
		yaml_path = self.get_parameter('yaml_path').get_parameter_value().string_value
		self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
  	    		

		self.spherical_projection = SphericalProjection(fov_up, fov_down, width, height)
		
		self.projection_transform = ProjectionSimTransform(self.spherical_projection)
		
		self.subscription = self.create_subscription(
			PointCloud2,
			'/velodyne_points',
			self.inference_callback,
			10
		)
		self.publisher = self.create_publisher(Image, '/segmentation_riunet', 10)
		self.publisher_stage2 = self.create_publisher(PointCloud2, '/segmentation_pc2_riunet', 10)
		self.bridge = CvBridge()
		self.model = RIUNet(in_channels=4, hidden_channels=(64, 128, 256, 512), n_classes=13) 
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
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
		
	def inference_callback(self, msg):
		pointcloud = structured_to_unstructured(pointcloud2_to_array(msg))
		pointcloud = pointcloud[:,:3]
		
		item = {
			'frame': pointcloud,
			'label': None,
			'mask': None
		}	
		
		item = self.projection_transform(item)
		item['frame'] = (item['frame'] * 255).astype(np.uint8) #duvida

		# Convert to float32 before passing to the model
		frame = item['frame'].astype(np.float32)
<<<<<<< HEAD

		# ldm = SemanticSegmentationSimLDM()
		# ldm.setup('predict')
		# epoch_steps = len(ldm.predict_dataloader())
		# n_epochs = 25
		# model = self.model
		# loss_fn =  CrossEntropyLoss(reduction='none')
		# viz_tfm = ldm.viz_tfm
		# total_steps = n_epochs*epoch_steps
		
		# loaded_model = SemanticSegmentationTask.load_from_checkpoint(
		# 	self.model_path, model=model, loss_fn=loss_fn, viz_tfm=viz_tfm, total_steps=total_steps
		# 	)
		
		# loaded_model.to(self.device)
		# loaded_model.eval()	

		loaded_model = torch.load(self.model_path, weights_only=True)
		self.model.load_state_dict(loaded_model['state_dict'], strict=False)
		self.model.to(self.device)
		self.model.eval()
=======
				
		loaded_model = torch.load(self.model_path)
		loaded_model.to(self.device)
		loaded_model.eval()
>>>>>>> bc74b92895b57e4829d6723593277c5640d670aa
		
		with torch.no_grad():
			frame = np.expand_dims(frame, 0)
			frame = torch.from_numpy(frame).to(self.device)
			frame = frame.permute(0, 3, 1, 2).float() # (N, H, W, C) -> (N, C, H, W)
			y_hat = self.model(frame).squeeze()
			argmax = torch.argmax(y_hat, dim=0)
			pred = np.array(argmax.cpu())
			#print(pred)
		item['label'] = pred
		colored_pred = self.projectionviz_transform(item)

		colored_pred_uint8 = colored_pred['label'].astype(np.uint8)
		
		inferred_frame_msg = self.bridge.cv2_to_imgmsg(colored_pred_uint8, encoding='passthrough')
		self.publisher.publish(inferred_frame_msg)

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
		inferred_point_cloud_msg = array_to_pointcloud2(frame_with_pred_structured)
		self.publisher_stage2.publish(inferred_point_cloud_msg)
		
def main(args=None):
	rclpy.init(args=args)
	riunet_inferencer = RIUNetInferencer()
	
	try:
		rclpy.spin(riunet_inferencer)
	except KeyboardInterrupt:
		pass
	finally:
		riunet_inferencer.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
    main()