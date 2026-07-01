import tkinter as tk
from tkinter import ttk
import math
import random

class HighRealismDrillSim:
    def __init__(self, root):
        self.root = root
        self.root.title("Künstliche Intelligenz im Maschinenbau - Digitale Bohrer-Simulation v4.0")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1E1E24")
        
        # --- PHYSIK- & SENSOR-VARIABLEN ---
        self.is_running = False
        self.rpm = tk.DoubleVar(value=1200)
        self.feed_rate = tk.DoubleVar(value=0.5) # mm/s
        self.tool_wear = tk.DoubleVar(value=10.0) # % Startverschleiß
        self.material_density = 1.0 # Multiplikator (z.B. Stahl = 1.0, Einschluss = 2.5)
        
        # Live-Sensoren
        self.live_torque = 0.0
        self.live_temp = 22.0 # Start bei Raumtemperatur
        self.live_vibration = 0.0
        self.anomaly_score = 0.0
        
        # Animations-Status
        self.drill_y = 100.0
        self.rotation_phase = 0.0
        self.particles = []
        self.has_inclusion = False
        
        # UI Style Setup
        self.setup_styles()
        
        # Layout aufbauen
        self.create_widgets()
        
        # Start des Haupt-Loops
        self.update_simulation()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1E1E24')
        style.configure('TLabel', background='#1E1E24', foreground='#E0E0E6', font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'), foreground='#00FFCC')
        style.configure('TScale', background='#1E1E24')
        style.configure('TCheckbutton', background='#1E1E24', foreground='#E0E0E6')

    def create_widgets(self):
        # Hauptcontainer (Gitter-Layout)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1) # Controls
        self.root.grid_columnconfigure(1, weight=2) # Canvas Animation
        self.root.grid_columnconfigure(2, weight=1) # KI Panel
        
        # ==========================================
        # 1. KONTROLLPANEL (LINKS)
        # ==========================================
        ctrl_frame = ttk.Frame(self.root, padding=15, style='TFrame')
        ctrl_frame.grid(row=0, column=0, sticky="nsew")
        
        ttk.Label(ctrl_frame, text="⚙️ MASCHINENPARAMETRE", style='Header.TLabel').pack(anchor="w", pady=10)
        
        # Start/Stopp Buttons
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill="x", pady=5)
        self.start_btn = tk.Button(btn_frame, text="START BOHRUNG", bg="#2ECC71", fg="white", font=('Segoe UI', 10, 'bold'), command=self.start_drill, relief="flat", padx=10, pady=5)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=2)
        self.stop_btn = tk.Button(btn_frame, text="NOT-AUS", bg="#E74C3C", fg="white", font=('Segoe UI', 10, 'bold'), command=self.stop_drill, relief="flat", padx=10, pady=5)
        self.stop_btn.pack(side="right", expand=True, fill="x", padx=2)
        
        # Slider für Drehzahl (RPM)
        ttk.Label(ctrl_frame, text="Soll-Drehzahl (RPM):").pack(anchor="w", pady=(15,2))
        self.rpm_scale = ttk.Scale(ctrl_frame, from_=0, to=3000, variable=self.rpm, orient="horizontal")
        self.rpm_scale.pack(fill="x")
        self.rpm_label = ttk.Label(ctrl_frame, text="1200 U/min")
        self.rpm_label.pack(anchor="e")
        
        # Slider für Vorschubgeschwindigkeit
        ttk.Label(ctrl_frame, text="Vorschubgeschwindigkeit (mm/s):").pack(anchor="w", pady=(15,2))
        self.feed_scale = ttk.Scale(ctrl_frame, from_=0.1, to=2.5, variable=self.feed_rate, orient="horizontal")
        self.feed_scale.pack(fill="x")
        self.feed_label = ttk.Label(ctrl_frame, text="0.5 mm/s")
        self.feed_label.pack(anchor="e")
        
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", pady=20)
        
        ttk.Label(ctrl_frame, text="⚠️ STÖRUNGS-SIMULATION", style='Header.TLabel').pack(anchor="w", pady=10)
        
        # Slider für Werkzeugverschleiß
        ttk.Label(ctrl_frame, text="Künstlicher Werkzeugverschleiß (%):").pack(anchor="w", pady=(5,2))
        self.wear_scale = ttk.Scale(ctrl_frame, from_=0, to=100, variable=self.tool_wear, orient="horizontal")
        self.wear_scale.pack(fill="x")
        self.wear_label = ttk.Label(ctrl_frame, text="10 %")
        self.wear_label.pack(anchor="e")
        
        # Checkbox für Materialeinschluss (z.B. gehärtete Stelle im Metall)
        self.incl_var = tk.BooleanVar(value=False)
        self.incl_cb = ttk.Checkbutton(ctrl_frame, text="Harter Materialeinschluss (Luft/Hartmetall)", variable=self.incl_var, command=self.toggle_inclusion)
        self.incl_cb.pack(anchor="w", pady=15)
        
        # Reset Button
        tk.Button(ctrl_frame, text="Simulation Zurücksetzen", bg="#34495E", fg="white", font=('Segoe UI', 9), command=self.reset_simulation, relief="flat").pack(fill="x", side="bottom", pady=10)

        # ==========================================
        # 2. DIGITAL TWIN CANVAS (MITTE)
        # ==========================================
        canvas_frame = ttk.Frame(self.root, padding=10)
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        
        ttk.Label(canvas_frame, text="🖥️ LIVE-ANIMATION (DIGITALER ZWILLING)", style='Header.TLabel').pack(anchor="w", pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg="#2D2D35", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")
        
        # ==========================================
        # 3. KI- & SENSORDASHBOARD (RECHTS)
        # ==========================================
        dash_frame = ttk.Frame(self.root, padding=15, style='TFrame')
        dash_frame.grid(row=0, column=2, sticky="nsew")
        
        ttk.Label(dash_frame, text="📊 TELEMETRIE / SENSORIK", style='Header.TLabel').pack(anchor="w", pady=10)
        
        # Sensor-Labels & Custom Progressbars für Telemetrie
        self.lbl_torque = ttk.Label(dash_frame, text="Drehmoment: 0.0 Nm", font=('Segoe UI', 11))
        self.lbl_torque.pack(anchor="w", pady=5)
        self.bar_torque = ttk.Progressbar(dash_frame, length=200, mode='determinate')
        self.bar_torque.pack(fill="x", pady=(0,10))
        
        self.lbl_temp = ttk.Label(dash_frame, text="Temperatur: 22.0 °C", font=('Segoe UI', 11))
        self.lbl_temp.pack(anchor="w", pady=5)
        self.bar_temp = ttk.Progressbar(dash_frame, length=200, mode='determinate')
        self.bar_temp.pack(fill="x", pady=(0,10))
        
        self.lbl_vib = ttk.Label(dash_frame, text="Vibration (g-Kraft): 0.0g", font=('Segoe UI', 11))
        self.lbl_vib.pack(anchor="w", pady=5)
        self.bar_vib = ttk.Progressbar(dash_frame, length=200, mode='determinate')
        self.bar_vib.pack(fill="x", pady=(0,10))
        
        ttk.Separator(dash_frame, orient="horizontal").pack(fill="x", pady=15)
        
        ttk.Label(dash_frame, text="🧠 KI-ANOMALIEERKENNUNG", style='Header.TLabel').pack(anchor="w", pady=10)
        
        # Anomalie-Score Anzeige
        self.lbl_anomaly = ttk.Label(dash_frame, text="Anomalie-Score: 0.0%", font=('Segoe UI', 12, 'bold'), foreground="#00FFCC")
        self.lbl_anomaly.pack(anchor="w", pady=5)
        self.bar_anomaly = ttk.Progressbar(dash_frame, length=200, mode='determinate')
        self.bar_anomaly.pack(fill="x", pady=(0,15))
        
        # KI Statustext-Box
        self.ai_status_box = tk.Label(dash_frame, text="SYSTEM BEREIT\nWarte auf Bohrprozess...", font=('Segoe UI', 11, 'bold'), bg="#2D2D35", fg="#8E44AD", h=4, relief="solid", bd=1)
        self.ai_status_box.pack(fill="x", pady=10)

    def start_drill(self):
        if self.rpm.get() > 50:
            self.is_running = True

    def stop_drill(self):
        self.is_running = False

    def toggle_inclusion(self):
        self.has_inclusion = self.incl_var.get()

    def reset_simulation(self):
        self.is_running = False
        self.drill_y = 100.0
        self.live_temp = 22.0
        self.live_torque = 0.0
        self.live_vibration = 0.0
        self.anomaly_score = 0.0
        self.incl_var.set(False)
        self.has_inclusion = False
        self.tool_wear.set(10.0)
        self.rpm.set(1200)
        self.feed_rate.set(0.5)
        self.particles = []
        self.ai_status_box.config(text="SYSTEM BEREIT\nWarte auf Bohrprozess...", bg="#2D2D35", fg="#8E44AD")

    # ==========================================
    # CORE RECHENEINHEIT (PHYSIK & KI MODEL)
    # ==========================================
    def update_simulation(self):
        # Update Text-Labels der Controls
        self.rpm_label.config(text=f"{int(self.rpm.get())} U/min")
        self.feed_label.config(text=f"{self.feed_rate.get():.2f} mm/s")
        self.wear_label.config(text=f"{int(self.tool_wear.get())} %")
        
        # 1. Physikalische Berechnungen, falls Maschine läuft
        current_rpm = self.rpm.get()
        current_feed = self.feed_rate.get()
        current_wear = self.tool_wear.get()
        
        # Überprüfung, ob der Bohrer das Werkstück berührt (Werkstück beginnt bei y=280)
        is_touching_material = (self.drill_y >= 260 and self.drill_y < 530)
        
        if is_touching_material and self.drill_y >= 380 and self.has_inclusion:
            self.material_density = 2.8 # Plötzlicher extrem harter Materialeinschluss
        elif is_touching_material:
            self.material_density = 1.0 # Normaler Baustahl
        else:
            self.material_density = 0.0 # Luft (kein Kontakt)

        if self.is_running and current_rpm > 50:
            # Vorschub-Bewegung nach unten realisieren
            self.drill_y += (current_feed * 0.4)
            if self.drill_y > 530: # Maximale Tiefe erreicht
                self.drill_y = 530
                self.is_running = False
            
            # Rotationsphase berechnen für visuelle Animation
            self.rotation_phase += (current_rpm / 600.0)
            
            # Realistische Physik-Formeln für die Sensoren
            if is_touching_material:
                # Drehmoment steigt mit Vorschub und Materialhärte, sinkt leicht mit zu hoher Drehzahl
                self.live_torque = (current_feed * 15.0 * self.material_density) + (current_wear * 0.15)
                # Vibration steigt massiv bei Materialänderung, Abnutzung oder kritischer Drehzahl/Resonanz
                base_vib = (current_rpm / 1500.0) + (current_feed * 2.0)
                wear_vib = (current_wear * 0.12)
                inclusion_vib = 8.5 if self.material_density > 2.0 else 0.0
                self.live_vibration = base_vib + wear_vib + inclusion_vib + random.uniform(-0.4, 0.4)
                
                # Temperatur akkumuliert sich durch Reibung
                heat_generation = (self.live_torque * current_rpm * 0.00005) + (current_wear * 0.02)
                self.live_temp += heat_generation - (self.live_temp - 22) * 0.01 # Reibungshitze vs Kühlung
            else:
                # Leerlauf
                self.live_torque = 0.5 + random.uniform(0, 0.2)
                self.live_vibration = (current_rpm / 2000.0) + random.uniform(0, 0.1)
                self.live_temp += (current_rpm * 0.001) - (self.live_temp - 22) * 0.02
        else:
            # Maschine steht still -> Abkühlung
            self.live_torque = 0.0
            self.live_vibration = 0.0
            self.live_temp -= (self.live_temp - 22) * 0.03
            if self.live_temp < 22: self.live_temp = 22

        # 2. SIMULIERTE KI-ANOMALIEERKENNUNG (Heuristik basierend auf ML-Logik)
        # Ein echtes ML-Modell würde Features extrahieren (z.B. RMS der Vibration, Frequenzverschiebungen)
        if is_touching_material:
            # Berechne Abweichung vom mathematischen Idealzustand (Neues Werkzeug, perfektes Material)
            ideal_torque = current_feed * 15.0 * 1.0
            ideal_vib = (current_rpm / 1500.0) + (current_feed * 2.0)
            
            torque_deviation = max(0.0, self.live_torque - ideal_torque)
            vib_deviation = max(0.0, self.live_vibration - ideal_vib)
            
            # Anomalie-Score skaliert hoch
            self.anomaly_score = (torque_deviation * 2.5) + (vib_deviation * 6.0) + (self.live_temp * 0.15 - 3.3)
            if self.anomaly_score > 100.0: self.anomaly_score = 100.0
            if self.anomaly_score < 0.0: self.anomaly_score = random.uniform(0.5, 3.0)
        else:
            self.anomaly_score = random.uniform(0.0, 1.5) # Rauschen im Leerlauf

        # KI-Status & Farbe updaten
        if self.anomaly_score < 25.0:
            self.ai_status_box.config(text="🟢 STATUS: OPTIMAL\nProzessparameter stabil.\nKeine Anomalien detektiert.", bg="#1E4620", fg="#2ECC71")
        elif self.anomaly_score < 60.0:
            self.ai_status_box.config(text="🟡 PRÄDIKTIVE WARTUNG\nErhöhter Verschleiß detektiert.\nWerkzeugtausch nach Schicht planen.", bg="#665215", fg="#F1C40F")
        elif self.anomaly_score < 85.0:
            self.ai_status_box.config(text="🟠 WARNUNG: ANOMALIE!\nUnerwartete Dichteänderung im\nMaterial oder kritische Vibration!", bg="#78341A", fg="#E67E22")
        else:
            self.ai_status_box.config(text="🔴 NOT-AUS EMPFOHLEN!\nKritischer Zustand! Drohender\nWerkzeugbruch oder Festfressen!", bg="#661C1C", fg="#E74C3C")

        # Telemetrie UI aktualisieren
        self.update_telemetry_ui()

        # 3. GRAPHISCHE ANIMATION REFRESH
        self.draw_digital_twin()

        # Rekursiver Loop mit ca. 40 FPS (25ms Intervall)
        self.root.after(25, self.update_simulation)

    def update_telemetry_ui(self):
        # Textupdates
        self.lbl_torque.config(text=f"Drehmoment: {self.live_torque:.1f} Nm")
        self.lbl_temp.config(text=f"Temperatur: {self.live_temp:.1f} °C")
        self.lbl_vib.config(text=f"Vibration (g-Kraft): {self.live_vibration:.2f}g")
        self.lbl_anomaly.config(text=f"Anomalie-Score: {self.anomaly_score:.1f}%")
        
        # Fortschrittsbalken-Skalierung (Schutz vor Überlauf)
        self.bar_torque['value'] = min(100, (self.live_torque / 60.0) * 100)
        self.bar_temp['value'] = min(100, (self.live_temp / 180.0) * 100)
        self.bar_vib['value'] = min(100, (self.live_vibration / 15.0) * 100)
        self.bar_anomaly['value'] = self.anomaly_score

    # ==========================================
    # GRAPHISCHE RENDERING ENGINE (CANVAS)
    # ==========================================
    def draw_digital_twin(self):
        self.canvas.delete("all")
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10: return # Fenster noch nicht voll geladen
        
        center_x = w // 2
        
        # 1. ZEICHNE WERKSTÜCK (MATERIAL)
        # Normales Material (Grau-Blau)
        self.canvas.create_rectangle(center_x - 120, 280, center_x + 120, 550, fill="#4A5568", outline="#718096", width=2)
        
        # Zeichne den harten Materialeinschluss, falls aktiviert
        if self.has_inclusion:
            self.canvas.create_rectangle(center_x - 118, 380, center_x + 118, 440, fill="#2C3E50", outline="#E74C3C", dash=(4,4))
            self.canvas.create_text(center_x + 190, 410, text="Störstelle\n(Gehärtetes Gefüge)", fill="#E74C3C", font=('Segoe UI', 9, 'bold'))
            self.canvas.create_line(center_x + 120, 410, center_x + 140, 410, fill="#E74C3C")
            
        # Zeichne bereits gebohrtes Loch
        if self.drill_y > 280:
            self.canvas.create_rectangle(center_x - 25, 278, center_x + 25, self.drill_y, fill="#1E1E24", outline="")
            
        # 2. ZEICHNE SPÄNE / PARTIKELEFFEKT (DYNAMISCH)
        if self.is_running and self.drill_y >= 275:
            # Partikel erzeugen basierend auf RPM und Vorschub
            num_particles = int((self.rpm.get() / 600.0) * (self.feed_rate.get() + 0.5))
            for _ in range(min(5, num_particles)):
                px = center_x + random.randint(-25, 25)
                py = self.drill_y + random.randint(-5, 2)
                vx = random.uniform(-6, 6)
                vy = random.uniform(-7, -2)
                # Farbe ändert sich bei glühendem Bohrer zu Funkenflug!
                color = "#FF9900" if self.live_temp > 90 else "#A0A0A0"
                self.particles.append([px, py, vx, vy, 1.0, color]) # x, y, vx, vy, alpha, color
                
        # Partikel updaten und zeichnen
        remaining_particles = []
        for p in self.particles:
            p[0] += p[1] # x + vx (Index-Fix: p[1] ist vx) -> p[0] ist x, p[1] ist y, etc.
            # Fix für saubere Vektorbewegung:
            p[0] += p[2] # x + vx
            p[1] += p[3] # y + vy
            p[3] += 0.5  # Schwerkraft-Effekt auf Späne
            p[4] -= 0.05 # Alpha-Fadeout
            if p[4] > 0 and p[1] < 560:
                remaining_particles.append(p)
                # Zeichnen
                self.canvas.create_oval(p[0]-2, p[1]-2, p[0]+2, p[1]+2, fill=p[5], outline="")
        self.particles = remaining_particles

        # 3. ZEICHNE BOHRERHALTERUNG & SPINDEL
        holder_y = self.drill_y - 140
        # Kleine Vibration visuell auf den Bohrer übertragen!
        vib_offset_x = random.uniform(-self.live_vibration*0.15, self.live_vibration*0.15) if self.is_running else 0
        
        # Spindelkopf
        self.canvas.create_rectangle(center_x - 45 + vib_offset_x, holder_y - 40, center_x + 45 + vib_offset_x, holder_y, fill="#555555", outline="#333333")
        self.canvas.create_rectangle(center_x - 30 + vib_offset_x, holder_y, center_x + 30 + vib_offset_x, holder_y + 40, fill="#777777", outline="#555555")

        # 4. ZEICHNE REALSITISCHE BOHRERSCHNECKE (MIT ROTATIONS-EFFEKT)
        drill_top = holder_y + 40
        drill_bottom = self.drill_y
        
        # Bohrer-Schaft Grundkörper
        # Thermische Farbveränderung (Bohrer fängt bei hoher Temp an zu glühen!)
        if self.live_temp < 60:
            drill_color = "#B0B0B5" # Normales Metall
        elif self.live_temp < 110:
            drill_color = "#D35400" # Leichtes Glühen (Dunkelorange)
        else:
            drill_color = "#E74C3C" # Kritisches Glühen (Hellrot)

        self.canvas.create_rectangle(center_x - 20 + vib_offset_x, drill_top, center_x + 20 + vib_offset_x, drill_bottom, fill=drill_color, outline="#7F8C8D")
        
        # Bohrerlippen / Wendel-Nuten animieren, um Rotation plastisch zu zeigen
        num_flutes = 7
        segment_h = (drill_bottom - drill_top) / num_flutes
        for i in range(num_flutes):
            seg_y_top = drill_top + (i * segment_h)
            seg_y_bottom = seg_y_top + segment_h
            
            # Sinuswelle simuliert die Helix-Rotation
            shift = math.sin(self.rotation_phase + (i * 1.2)) * 18
            
            # Zeichne geschwungene Schneidkanten
            self.canvas.create_line(center_x + vib_offset_x - 20, seg_y_top, center_x + vib_offset_x + shift, (seg_y_top + seg_y_bottom)/2, center_x + vib_offset_x + 20, seg_y_bottom, fill="#34495E", width=2, smooth=True)

        # Konische Bohrerspitze
        tip_color = "#FF3300" if self.live_temp > 100 else ("#D35400" if self.live_temp > 60 else "#95A5A6")
        self.canvas.create_polygon(
            center_x - 20 + vib_offset_x, drill_bottom,
            center_x + 20 + vib_offset_x, drill_bottom,
            center_x + vib_offset_x, drill_bottom + 15,
            fill=tip_color, outline="#7F8C8D"
        )
        
        # Bemaßungen & Live-Tiefenanzeige daneben rendern
        depth_mm = max(0.0, (self.drill_y - 260) * 0.5) # Skaliert auf mm
        self.canvas.create_text(center_x - 150, self.drill_y, text=f"Z-Tiefe: {depth_mm:.2f} mm", fill="#00FFCC", font=('Consolas', 10, 'bold'), anchor="e")
        self.canvas.create_line(center_x - 140, self.drill_y, center_x - 30 + vib_offset_x, self.drill_y, fill="#00FFCC", dash=(2,2))

if __name__ == "__main__":
    root = tk.Tk()
    app = HighRealismDrillSim(root)
    root.mainloop()
