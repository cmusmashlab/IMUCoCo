//
//  SocketClient.swift
//  P2P_Client_New
//
//  Created by 荒川陸 on 2022/03/22.
//

import Foundation
import Network

class SocketClient {

    let host: NWEndpoint.Host
    let port: NWEndpoint.Port
    var connection: NWConnection!
    let queue = DispatchQueue.global()
    
    init(ip: String, portInt: UInt16, completion: @escaping (Bool) -> Void) {
        host = NWEndpoint.Host(ip)
        port = NWEndpoint.Port(integerLiteral: portInt)
        startNewConnection(completion: completion)
        send(text: "client initialized")
    }
    
    private func startNewConnection(completion: @escaping (Bool) -> Void) {
        
        connection = NWConnection(host: host, port: port, using: .udp)
        connection.stateUpdateHandler = { (newState) in
            switch newState {
            case .ready:
                completion(true)
            default:
                completion(false)
            }
        }
        connection.start(queue: queue)
        receive(on: connection)
    }
    
    func stop() {
        send(text: "stop") { [self] in
            connection.cancel()
        }
    }
    
    func restart(completion: @escaping (Bool) -> Void) {
        startNewConnection(completion: completion)
    }
    
    func receive(on connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 0, maximumLength: Int(UInt32.max)) { [weak self] (data, _, _, error) in
            if let data = data {
                let text = String(data: data, encoding: .utf8)!
                print("received \(text)")
            } else {
                print("\(#function), Received data is nil")
            }
        }
    }
    
    func send(text: String, _ completion: (() -> Void)? = nil) {
        let data = (text).data(using: .utf8)!

        connection.send(content: data, completion: .contentProcessed { [unowned self] (error) in
            if let error = error {
                print("\(#function), \(error)")
            } else {
                // print("send \(text)")
                if let completion = completion {
                    completion()
                }
            }
        })
    }
    
    func send(data: Data, _ completion: (() -> Void)? = nil) {
        connection.send(content: data, completion: .contentProcessed { [unowned self] (error) in
            if let error = error {
                print("\(#function), \(error)")
            } else {
                // print("send \(text)")
                if let completion = completion {
                    completion()
                }
            }
        })
    }
}
