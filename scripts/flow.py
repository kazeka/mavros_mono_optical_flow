#!/usr/bin/env python

import cv2 as cv
import cv_bridge
import dronekit
import numpy as np
import rospy
import sensor_msgs.msg

from optical_flow.msg import OpticalFlow


class OpticalFlowNode:
    def __init__(self):
        rospy.init_node('optical_flow')

        self.vehicle_connection_string = rospy.get_param('~connection_string', default='/dev/ttyTHS2')
        self.vehicle_baudrate = rospy.get_param('~baudrate', default=57600)

        rospy.loginfo('Connecting to {} at {} baud.'.format(self.vehicle_connection_string, self.vehicle_baudrate))
        self.vehicle = dronekit.connect(self.vehicle_connection_string, wait_ready=True, baud=self.vehicle_baudrate)
        rospy.loginfo('Connected.')

        self.sensor_id = 64
        self.input_topic = rospy.get_param('~input_topic', default='/flir/left/image_raw')
        self.output_topic = rospy.get_param('~output_topic', default='/optical_flow/flow')
        self.prev_img = None
        self.prev_features = None
        self.img = None
        self.features = None

        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Hanning window for Phase correlation
        self.array = cv.createHanningWindow((1296, 1024), cv.CV_32F)

        self.img_sub = rospy.Subscriber(self.input_topic, sensor_msgs.msg.Image, callback=self.image_callback, queue_size=1)
        rospy.loginfo('Listening to {}.'.format(self.input_topic))

        self.optical_flow_pub = rospy.Publisher('optical_flow', OpticalFlow, queue_size=1)
        rospy.loginfo('Writing to {}.'.format(self.output_topic))


    def image_callback(self, img_msg):
        img = cv_bridge.CvBridge().img_msg_to_cv2(img_msg)
        if self.prev_img is None:
            self.prev_img = img.copy()
            self.prev_features = cv.goodFeaturesToTrack(self.prev_img, mask=None, **self.feature_params)
        else:
            self.img_usec = int(img_msg.header.stamp.secs*1000000 + img_msg.header.stamp.nsecs/1000)
            self.img = img.copy()
            # pix_x, pix_y, confidence = self.calculate_pix()
            pix_x, pix_y, confidence = self.calculate_phase_corr()
            self.send_mavlink_msg(pix_x, pix_y, confidence)
            self.send_ros_msg(pix_x, pix_y, confidence)
            self.prev_img = img.copy()
    

    def calculate_phase_corr(self):
        """
            Optical flow using phase correlation analogous to OpenMV sensor
        """
        shift, response = cv.phaseCorrelate(self.prev_img.astype('float32'), self.img.astype('float32'), window=self.array)
        return shift[0], shift[1], response


    def calculate_pix(self):
        """
            Compute average optical flow in x and y in pixels
            Confidence is the number of total tracked features with good status==1
        """
        self.features, self.status, self.error = cv.calcOpticalFlowPyrLK(
            self.prev_img, self.img, self.prev_features, None, **self.lk_params)
        if self.status.any():
            # TODO: compare mean and mode for stability
            pix_x = int((self.features[:,:,0][self.status==1]-self.prev_features[:,:,0][self.status==1]).mean())
            pix_y = int((self.features[:,:,1][self.status==1]-self.prev_features[:,:,1][self.status==1]).mean())
            confidence = int(self.status.sum() / len(self.status) * 255.)
        else:
            pix_x, pix_y, confidence = 0, 0, 0
        # re-track new features to avoid cheking how many old features are still valid
        corners = cv.goodFeaturesToTrack(self.img, mask=None, **self.feature_params)
        if corners is not None:
            self.prev_features = corners
        rospy.logdebug_throttle(.5, 'OF: {} {} {}'.format(pix_x, pix_y, confidence))
        return pix_x, pix_y, confidence


    def send_ros_msg(self, pix_x, pix_y, confidence):
        msg = OpticalFlow()
        msg.ground_distance = -1
        msg.flow_x = pix_x
        msg.flow_y = pix_y
        msg.velocity_x = 0.
        msg.velocity_y = 0.
        msg.quality = confidence
        self.optical_flow_pub.publish(msg)


    def send_mavlink_msg(self, pix_x, pix_y, confidence):
        msg = self.vehicle.message_factory.optical_flow_encode(
            self.img_usec,
            self.sensor_id,
            pix_x, pix_y,
            0., 0.,
            confidence,
            -1.)
        self.vehicle.send_mavlink(msg)


if __name__ == '__main__':
    of = OpticalFlowNode()
    rospy.spin()