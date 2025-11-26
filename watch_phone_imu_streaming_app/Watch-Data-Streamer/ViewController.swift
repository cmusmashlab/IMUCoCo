//
//  ViewController.swift
//  Watch-Data-Streamer
//
//  Created by Vicky Liu on 2/25/22.
//  Updated by Haozhe Zhou 10/03/2025
//

import UIKit
import CoreMotion
import WatchConnectivity

class ViewController: UIViewController {
    

    @IBOutlet weak var statusLabel: UILabel!
    @IBOutlet weak var watchIMULabel: UILabel!
    @IBOutlet weak var phoneIMULabel: UILabel!
    @IBOutlet weak var startButton: UIButton!
    @IBOutlet weak var stopButton: UIButton!

    // user ID
    var userId: String!
    @IBOutlet weak var userIdTextField: UITextField!
    
    // watch motion frequency
    let nToMeasureFrequency = 50
    let expectedFrequency: Double = 50.0
    let sampelsPerPacket = 1 // previously is 10;
    let samplesPerPacketWatch = 1
    
    var watchCnt = 0
    var watchPrevTime: TimeInterval = NSDate().timeIntervalSince1970
    var phoneCnt = 0
    var phonePrevTime: TimeInterval = NSDate().timeIntervalSince1970
    
    // phone motion
    let motionManager = CMMotionManager()
    let queue = OperationQueue()
    var imuSendEnabled = false
    var phoneIMUTextList: [String] = []
    @IBOutlet weak var imuSwitch: UISwitch!
    
    // socket
    var socketClient: SocketClient?
    @IBOutlet weak var socketIPTextField: UITextField!
    @IBOutlet weak var socketPortTextField: UITextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        if (WCSession.isSupported()) {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
        
        if let savedIP = UserDefaults.standard.string(forKey: "IP") {
            socketIPTextField.text = savedIP
        } else {
            socketIPTextField.text = "0.0.0.0"
        }
        
        if let savedPort = UserDefaults.standard.string(forKey: "Port") {
            socketPortTextField.text = savedPort
        } else {
            socketPortTextField.text = "8000"
        }
        
        if let _userId = UserDefaults.standard.string(forKey: "userID") {
            userId = _userId
        } else {
            userId = "defaultUser"
        }
        userIdTextField.text = userId
        
        self.hideKeyboardWhenTapped()
        
        statusLabel.text = "Watch should be paired."
        startButton.isEnabled = false
        stopButton.isEnabled = false
        imuSwitch.isOn = false
    }
    
    @IBAction func createSocketConnection() {
        let ip = socketIPTextField.text as String? ?? "0.0.0.0"
        let portText = socketPortTextField.text ?? "8000"
        UserDefaults.standard.set(ip, forKey: "IP")
        UserDefaults.standard.set(portText, forKey: "Port")
    
        socketClient = SocketClient(ip: ip, portInt: UInt16(portText) ?? 8000) { (status) in
            print("socket status: \(status)")
        }
    }
    
    @IBAction func stopSocketConnection() {
        guard let socketClient = socketClient else {
            return
        }
        socketClient.stop()
    }
    
    @IBAction func pairWatch() {

        if WCSession.default.isReachable {
            do {
                watchCnt = 0
                try WCSession.default.updateApplicationContext(["command": "pair"])
                
                statusLabel.text = "See watch if it is paired."
                startButton.isEnabled = true
                stopButton.isEnabled = false
            }
            catch {
                print(error)
            }
        } else {
            statusLabel.text = "Watch is not reachable"
        }
    }
    
    @IBAction func start(_ sender: Any) {
        
        userId = userIdTextField.text ?? "defaultUser"
        UserDefaults.standard.set(userId, forKey: "userID")
        
        if (WCSession.default.isReachable) {
            do {
                try WCSession.default.updateApplicationContext(["command": "start"])
                startingOperations()
            }
            catch {
                print(error)
            }
        } else {
            statusLabel.text = "Watch is not reachable"
        }
        
        startPhoneIMU()
    }
    
    private func startPhoneIMU() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / expectedFrequency
            motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: queue) { [weak self] (data, error) in
                guard let self = self, let data = data, self.imuSendEnabled else { return }

                let text = "\(NSDate().timeIntervalSince1970) \(data.userAcceleration.x) \(data.userAcceleration.y) \(data.userAcceleration.z) \(data.gravity.x) \(data.gravity.y) \(data.gravity.z) \(data.rotationRate.x) \(data.rotationRate.y) \(data.rotationRate.z) \(data.attitude.quaternion.x) \(data.attitude.quaternion.y) \(data.attitude.quaternion.z) \(data.attitude.quaternion.w)\n"
                
                self.phoneIMUTextList.append(text)
                phoneCnt += 1
                if (phoneCnt % sampelsPerPacket == 0) {
                    let message = self.phoneIMUTextList.joined(separator: "&")
                    self.phoneIMUTextList = []
                    if let socketClient = self.socketClient, socketClient.connection.state == .ready {
                        socketClient.send(text: "\(userId!);phone;motion:" + message)
                    }
                }
                
                if phoneCnt % nToMeasureFrequency == 0 {
                    let currentTime = NSDate().timeIntervalSince1970
                    let timeDiff = (currentTime - phonePrevTime) as Double
                    phonePrevTime = currentTime
                    let phoneMeasuredFrequency = 1.0 / timeDiff * Double(nToMeasureFrequency)
                    DispatchQueue.main.async {
                        self.phoneIMULabel.text = "Phone IMU: \(self.phoneCnt) data -- \(round(100 * phoneMeasuredFrequency) / 100) [Hz]"
                    }
                }
            }
        }
    }
    
    @IBAction func stop(_ sender: Any) {
        
        if (WCSession.default.isReachable) {
            do {
                try WCSession.default.updateApplicationContext(["command": "stop"])
                stoppingOperations()
            }
            catch {
                print(error)
            }
        } else {
            statusLabel.text = "Watch is not reachable"
        }
        
        if motionManager.isDeviceMotionAvailable {
            motionManager.stopDeviceMotionUpdates()
            print ("Stop motion recording")
        }
    }
    
    @IBAction func imuSwitchChanged(_ sender: UISwitch) {
        imuSendEnabled = sender.isOn
    }
    
    private func startingOperations() {
        
        DispatchQueue.main.async {
            self.statusLabel.text = "Recording"
            self.startButton.isEnabled = false
            self.stopButton.isEnabled = true
        }
    }
    
    private func stoppingOperations() {
        
        DispatchQueue.main.async {
            self.statusLabel.text = "Stopped Recording"
            self.startButton.isEnabled = true
            self.stopButton.isEnabled = false
        }
    }
}


extension ViewController: WCSessionDelegate {
    
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any] = [:]) {
        if let motionData = message["motionData"] as? String {
            print ("Received motion data")
            
            if let socketClient = self.socketClient {
                if socketClient.connection.state == .ready {
                    socketClient.send(text: "\(userId!);watch;motion:" + motionData)
                }
            }
            
            watchCnt += self.samplesPerPacketWatch  // should match with the watch setting
            if watchCnt % nToMeasureFrequency == 0 {
                let currentTime = NSDate().timeIntervalSince1970
                let timeDiff = (currentTime - watchPrevTime) as Double
                watchPrevTime = currentTime
                let watchMeasuredFrequency = 1.0 / timeDiff * Double(nToMeasureFrequency)
                DispatchQueue.main.async {
                    self.watchIMULabel.text = "Watch IMU: \(self.watchCnt) data -- \(round(100 * watchMeasuredFrequency) / 100) [Hz]"
                }
            }
        }
        
        if let audioData = message["audioData"] as? Data {
            print ("Received audio data")
            if let socketClient = self.socketClient {
                if socketClient.connection.state == .ready {
                    socketClient.send(data: audioData)
                }
            }
        }
        
        // logging to know if the motion data is saved
        if let status = message["command"] as? String {
            if status == "start" {
                startingOperations()
            } else if status == "stop" {
                stoppingOperations()
            }
        }
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        
    }
}


extension UIViewController {

    @objc func hideKeyboardWhenTapped() {
        let tap: UITapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(UIViewController.dismissKeyboard))
        tap.cancelsTouchesInView = false
        view.addGestureRecognizer(tap)
    }

    @objc func dismissKeyboard() {
        view.endEditing(true)
    }
}

extension ViewController: UITextFieldDelegate {
    
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        return true
    }
}
