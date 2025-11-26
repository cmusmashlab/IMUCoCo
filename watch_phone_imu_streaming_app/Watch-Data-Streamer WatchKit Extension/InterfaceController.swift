//
//  InterfaceController.swift
//  Watch-Data-Streamer WatchKit Extension
//
//  Created by Vicky Liu on 2/25/22.
//

import WatchKit
import Foundation
import WatchConnectivity
import UIKit
import AVFoundation


class InterfaceController: WKInterfaceController {
    
    @IBOutlet weak var statusLabel: WKInterfaceLabel!
    @IBOutlet weak var startButton: WKInterfaceButton!
    @IBOutlet weak var stopButton: WKInterfaceButton!
    @IBOutlet weak var audioEnableSwitch: WKInterfaceSwitch!

    override func awake(withContext context: Any?) {
        // Configure interface objects here.
        statusLabel.setText("Need Pairing")
        startButton.setEnabled(false)
        stopButton.setEnabled(false)
        
        if let savedAudioRecordingEnabled = UserDefaults.standard.object(forKey: "audioRecordingEnabled") {
            audioEnableSwitch.setOn(savedAudioRecordingEnabled as! Bool)
        } else {
            UserDefaults.standard.set(true, forKey: "audioRecordingEnabled")
            audioEnableSwitch.setOn(true)
        }
    }
    
    override func willActivate() {
        // This method is called when watch view controller is about to be visible to user
        print("will activate!!!")
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
            print("is reachable ", session.isReachable)
        }
    }
    
    override func didDeactivate() {
        // This method is called when watch view controller is no longer visible
    }
    
    private func start() {
        WorkoutManager.shared.requestHealthKitAuthorization()
        WorkoutManager.shared.startWorkout()
        MotionManager.shared.startRecording()
        if let savedAudioRecordingEnabled = UserDefaults.standard.object(forKey: "audioRecordingEnabled") as? Bool {
            if savedAudioRecordingEnabled {
                AudioManager.shared.startRecording()
            }
        }
        statusLabel.setText("Recording")
        startButton.setEnabled(false)
        stopButton.setEnabled(true)
        audioEnableSwitch.setEnabled(false)
    }
    
    private func stop() {
        WorkoutManager.shared.endWorkout()
        MotionManager.shared.endRecording()
        if let savedAudioRecordingEnabled = UserDefaults.standard.object(forKey: "audioRecordingEnabled") as? Bool {
            if savedAudioRecordingEnabled {
                AudioManager.shared.endRecording()
            }
        }
        statusLabel.setText("Stopped Recording")
        startButton.setEnabled(true)
        stopButton.setEnabled(false)
        audioEnableSwitch.setEnabled(true)
    }
    
    @IBAction func startButtonPressed() {
        WCSession.default.sendMessage(["command": "start"], replyHandler: nil)
        start()
    }
    
    @IBAction func stopButtonPressed() {
        WCSession.default.sendMessage(["command": "stop"], replyHandler: nil)
        stop()
    }
    
    @IBAction func switchChanged(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "audioRecordingEnabled")
    }
    
}

extension InterfaceController: WCSessionDelegate {
    
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        
    }
    
    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String : Any]) {
        
        if let command = applicationContext["command"] as? String {
            if command == "start" {
                start()
            } else if command == "stop" {
                stop()
            } else if command == "pair" {
                statusLabel.setText("Paired. Ready to record.")
                startButton.setEnabled(true)
                stopButton.setEnabled(false)
            }
        }
    }
}
