import numpy as np
import random

class HospitalEnv:
    def __init__(self):
        self.max_icu        = 10
        self.max_gen_beds   = 20
        self.max_doctors    = 8
        self.max_ambulances = 5

        self.history = []   # track metrics over time

    def reset(self):
        self.icu_beds       = self.max_icu
        self.gen_beds       = self.max_gen_beds
        self.doctors        = self.max_doctors
        self.ambulances     = self.max_ambulances

        self.critical_patients  = random.randint(1, 3)
        self.moderate_patients  = random.randint(2, 5)
        self.mild_patients      = random.randint(1, 4)

        self.timestep           = 0
        self.patients_treated   = 0
        self.patients_died      = 0
        self.total_wait_time    = 0
        self.history            = []
        return self._get_state()

    def _get_state(self):
        def level(val, max_val):
            r = val / max_val
            if r < 0.33:  return 0
            elif r < 0.66: return 1
            else:          return 2

        return (
            level(self.icu_beds,       self.max_icu),
            level(self.gen_beds,       self.max_gen_beds),
            level(self.doctors,        self.max_doctors),
            level(self.ambulances,     self.max_ambulances),
            min(self.critical_patients,  2),
            min(self.moderate_patients // 2, 2),
        )

    def get_full_state(self):
        """Returns full numeric state for the frontend."""
        return {
            "icu_beds":          self.icu_beds,
            "max_icu":           self.max_icu,
            "gen_beds":          self.gen_beds,
            "max_gen_beds":      self.max_gen_beds,
            "doctors":           self.doctors,
            "max_doctors":       self.max_doctors,
            "ambulances":        self.ambulances,
            "max_ambulances":    self.max_ambulances,
            "critical_patients": self.critical_patients,
            "moderate_patients": self.moderate_patients,
            "mild_patients":     self.mild_patients,
            "patients_treated":  self.patients_treated,
            "patients_died":     self.patients_died,
            "timestep":          self.timestep,
        }

    def step(self, action):
        reward = 0
        event  = ""

        # --- Action effects ---
        if action == 0:   # Assign ICU to critical
            if self.icu_beds > 0 and self.critical_patients > 0:
                self.icu_beds          -= 1
                self.doctors           -= 1 if self.doctors > 0 else 0
                self.critical_patients -= 1
                self.patients_treated  += 1
                reward = +50
                event  = "✅ Critical patient assigned ICU bed"
            else:
                reward = -20
                event  = "⚠️ ICU assignment failed — no bed or no critical patient"

        elif action == 1: # Assign general bed to moderate
            if self.gen_beds > 0 and self.moderate_patients > 0:
                self.gen_beds          -= 1
                self.moderate_patients -= 1
                self.patients_treated  += 1
                reward = +20
                event  = "✅ Moderate patient assigned general bed"
            else:
                reward = -10
                event  = "⚠️ General bed assignment failed"

        elif action == 2: # Dispatch ambulance
            if self.ambulances > 0:
                self.ambulances -= 1
                reward = +15
                event  = "🚑 Ambulance dispatched"
            else:
                reward = -30
                event  = "❌ No ambulance available!"

        elif action == 3: # Assign doctor to mild patient
            if self.doctors > 0 and self.mild_patients > 0:
                self.doctors      -= 1
                self.mild_patients -= 1
                self.patients_treated += 1
                reward = +10
                event  = "✅ Mild patient treated by doctor"
            else:
                reward = -5
                event  = "⚠️ Doctor assignment failed"

        elif action == 4: # Queue patient (wait)
            reward = -10
            self.total_wait_time += 1
            event  = "⏳ Patient queued — waiting"

        # --- Critical patient penalty if unhandled ---
        if self.critical_patients > 2:
            reward         -= 50
            self.patients_died += 1
            event          += " | ☠️ Critical patient deteriorated!"

        # --- New arrivals each timestep ---
        new_critical  = random.randint(0, 2)
        new_moderate  = random.randint(0, 3)
        new_mild      = random.randint(0, 2)
        self.critical_patients  = min(self.critical_patients + new_critical,  10)
        self.moderate_patients  = min(self.moderate_patients + new_moderate,  15)
        self.mild_patients      = min(self.mild_patients + new_mild,           10)

        # --- Resource recovery ---
        self.doctors    = min(self.doctors + 1,    self.max_doctors)
        self.ambulances = min(self.ambulances + 1, self.max_ambulances)

        self.timestep += 1
        done = self.timestep >= 100

        # --- Log history ---
        self.history.append({
            "step":             self.timestep,
            "reward":           reward,
            "patients_treated": self.patients_treated,
            "patients_died":    self.patients_died,
            "icu_beds":         self.icu_beds,
            "event":            event,
        })

        return self._get_state(), reward, done, event