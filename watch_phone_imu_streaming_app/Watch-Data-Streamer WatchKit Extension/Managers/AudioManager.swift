//
//  AudioManager.swift
//  Watch-Data-Streamer WatchKit Extension
//
//  Created by Vicky Liu on 2/28/22.
//

import AVFoundation
import WatchConnectivity

class AudioManager: NSObject {

    static let shared = AudioManager()
    let audioSession = AVAudioSession.sharedInstance()
    let audioEngine = AVAudioEngine()

    private override init() {
        super.init()
        setupAudioSession()
    }
    
    private func setupAudioSession() {
        // this is helpful: https://arvindhsukumar.medium.com/using-avaudioengine-to-record-compress-and-stream-audio-on-ios-48dfee09fde4

        do {
            try audioSession.setCategory(.playAndRecord, mode: .default)  // for matching with the async record app
            try audioSession.setActive(true) // for continuing the MOTION recording.
            audioSession.requestRecordPermission() { _ in return }

            let audioInputNode = audioEngine.inputNode
            let audioFormat = audioInputNode.inputFormat(forBus: 0)
            print("default format: \(audioFormat)")
            audioInputNode.installTap(onBus: 0, bufferSize: 1024*8, format: audioFormat) { buffer, time in
                // the max frequency for watch connectivity seems 10Hz, so don't set bufferSize less than 4800.
                // large batch size is preferred for better connectivity (1024 * 8 seems best on 10/31/2024)
                self.streamBufferToPhone(buffer: buffer)
            }
            audioEngine.prepare()
        } catch {
            print("Error setting up audio capture: \(error.localizedDescription)")
        }
    }
    
    func startRecording() {
        do {
            try audioEngine.start()
        } catch {
            print("Error starting audio capture: \(error.localizedDescription)")
        }
    }
    
    func endRecording() {
        audioEngine.stop()
    }

    private func streamBufferToPhone(buffer: AVAudioPCMBuffer) {
        // Stream audio data to the iPhone using WCSession
        // Apple bans low-level network use quite strictly: https://developer.apple.com/documentation/technotes/tn3135-low-level-networking-on-watchos
        // streaming audio to the watch is ok but the opposite did not work (2024/10/31)
        if let data = convertBufferToData(buffer: buffer) {
            if (WCSession.default.isReachable) {
                WCSession.default.sendMessage(["audioData": data], replyHandler: nil)
                print ("Audio data transferred")
            } else {
                print ("WCSession is not activated")
            }
        }
    }
    
    private func convertBufferToData(buffer: AVAudioPCMBuffer) -> Data? {
        let channelCount = 1
        let channelData = buffer.floatChannelData!
        let length = UInt32(buffer.frameLength) * UInt32(channelCount)
        let data = Data(bytes: channelData[0], count: Int(length) * MemoryLayout<Float>.size)
        return data
    }

}

