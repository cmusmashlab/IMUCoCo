//
//  WorkoutManager.swift
//  SensorLogger WatchKit Extension
//
//  Created by taeyoung yeon on 9/17/24.
//  Modified by riku arakawa on 2/24/25
//


import Foundation
import HealthKit

class WorkoutManager: NSObject {
    
    static let shared = WorkoutManager()
    private var workoutSession: HKWorkoutSession?
    private var sessionBuilder: HKLiveWorkoutBuilder?
    private var healthStore = HKHealthStore()
    private(set) var isWorkoutActive: Bool = false
    
    private override init() {  // make it private
        super.init()
    }
    
    func requestHealthKitAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("HealthKit is not available on this device.")
            return
        }
        
        let typesToShare: Set = [
            HKObjectType.workoutType()
        ]
        
        let typesToRead: Set = [
            HKObjectType.workoutType(),
            // HKObjectType.quantityType(forIdentifier: .heartRate)!
        ]
        
        healthStore.requestAuthorization(toShare: typesToShare, read: typesToRead) { success, error in
            if success {
                print("HealthKit authorization granted.")
            } else {
                print("HealthKit authorization denied: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
    }

    func startWorkout() {
        if isWorkoutActive {
            print("Already running workout mode")
            return
        }

        let configuration = HKWorkoutConfiguration()
        configuration.activityType = .other
        configuration.locationType = .unknown

        do {
            workoutSession = try HKWorkoutSession(healthStore: healthStore, configuration: configuration)
            sessionBuilder = workoutSession?.associatedWorkoutBuilder()
            workoutSession?.delegate = self
            sessionBuilder?.delegate = self

            workoutSession?.startActivity(with: Date())
            isWorkoutActive = true
            sessionBuilder?.beginCollection(withStart: Date(), completion: { (success, error) in
                if let error = error {
                    print("Error starting workout session: \(error.localizedDescription)")
                }
            })
        } catch {
            print("Failed to start workout session: \(error.localizedDescription)")
        }
    }

    func endWorkout() {
        sessionBuilder?.endCollection(withEnd: Date()) { (success, error) in
            if let error = error {
                print("Error ending workout session: \(error.localizedDescription)")
            } else {
                self.sessionBuilder?.finishWorkout { (workout, error) in
                    if let error = error {
                        print("Error finishing workout: \(error.localizedDescription)")
                    } else {
                        print("Workout successfully finished.")
                        self.workoutSession?.end()
                        self.isWorkoutActive = true
                    }
                }
            }
        }
    }

}

extension WorkoutManager: HKWorkoutSessionDelegate {

    func workoutSession(_ workoutSession: HKWorkoutSession, didChangeTo toState: HKWorkoutSessionState, from fromState: HKWorkoutSessionState, date: Date) {
        // Handle state changes if necessary
        print("Workout state changed from \(fromState.rawValue) to \(toState.rawValue) at \(date)")
        if toState == .ended {
            print("Workout session ended. Cleaning up...")
            self.workoutSession = nil
            self.sessionBuilder = nil
            isWorkoutActive = false
        }
    }

    func workoutSession(_ workoutSession: HKWorkoutSession, didFailWithError error: Error) {
        print("Workout session failed: \(error.localizedDescription)")
        isWorkoutActive = false
    }
}

extension WorkoutManager: HKLiveWorkoutBuilderDelegate {

    func workoutBuilderDidCollectEvent(_ workoutBuilder: HKLiveWorkoutBuilder) {
        // Handle events if necessary
    }
    
    func workoutBuilder(_ workoutBuilder: HKLiveWorkoutBuilder, didCollectDataOf collectedTypes: Set<HKSampleType>) {
        
    }
}
