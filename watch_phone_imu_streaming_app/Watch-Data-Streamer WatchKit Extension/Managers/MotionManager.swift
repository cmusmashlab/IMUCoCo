//
//  MotionManager.swift
//  Watch-Data-Streamer WatchKit Extension
//
//  Created by Vicky Liu on 2/25/22.
//  Updated by Haozhe Zhou 10/03/2025
//

import CoreMotion
import WatchConnectivity

class MotionManager {
    
    static let shared = MotionManager()
    let motionManager = CMMotionManager()
    let queue = OperationQueue()
    let sampleIntervalSec = 1.0 / 50
    let sampelsPerPacket = 1 // previously is 10;
    var textList: [String] = []
    
    
    private init() {
        queue.maxConcurrentOperationCount = 1
        queue.name = "MotionManagerQueue"
        if !(motionManager.isDeviceMotionAvailable) {
            print ("Device motion is unavailable")
            return
        }
        motionManager.deviceMotionUpdateInterval = sampleIntervalSec
    }
  
    func startRecording() {
        print("Start IMU updates")
        var cnt = 0

        motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: queue) { (data, error) in
            guard error == nil else { return }
            if let data = data {
//                let text = "\(NSDate().timeIntervalSince1970) \(data.userAcceleration.x) \(data.userAcceleration.y) \(data.userAcceleration.z) \(data.gravity.x) \(data.gravity.y) \(data.gravity.z) \(data.rotationRate.x) \(data.rotationRate.y) \(data.rotationRate.z) \(data.magneticField.field.x) \(data.magneticField.field.y) \(data.magneticField.field.z) \(data.attitude.roll) \(data.attitude.pitch) \(data.attitude.yaw) \(data.attitude.quaternion.x) \(data.attitude.quaternion.y) \(data.attitude.quaternion.z) \(data.attitude.quaternion.w) \(data.timestamp) \n"
                
                // use smaller amount for PrISM + IMUCoCo
                let text = "\(NSDate().timeIntervalSince1970) \(data.userAcceleration.x) \(data.userAcceleration.y) \(data.userAcceleration.z) \(data.gravity.x) \(data.gravity.y) \(data.gravity.z) \(data.rotationRate.x) \(data.rotationRate.y) \(data.rotationRate.z) \(data.attitude.quaternion.x) \(data.attitude.quaternion.y) \(data.attitude.quaternion.z) \(data.attitude.quaternion.w)\n"
                
                self.textList.append(text)
                cnt += 1
                if (cnt == Int(self.sampelsPerPacket)) {
                    cnt = 0
                    let message = self.textList.joined(separator: "&")
                    self.textList = []
                    if (WCSession.default.isReachable) {
                        WCSession.default.sendMessage(["motionData": message], replyHandler: nil)
                        // print ("Motion data transferred")
                    } else {
                        print ("WCSession is not activated")
                    }
                }
            }
        }
    }
    
    func endRecording() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.stopDeviceMotionUpdates()
            print ("Stop motion recording")
        }
    }
}
