\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[ngerman]{babel}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{geometry}
\usepackage{enumitem}
\usepackage{hyperref}

\geometry{margin=2.5cm}

% Farben für die didaktische Gestaltung
\definecolor{darkblue}{rgb}{0.0, 0.0, 0.5}
\definecolor{deepgreen}{rgb}{0.0, 0.5, 0.0}

\title{\textbf{Schulungsunterlage: KI-Zwilling Bohrsystem v21.6}}
\author{Industrie 4.0 Kompetenzzentrum | Maschinenbau \& KI-Inferenz}
\date{19. Januar 2026}

\begin{document}

\maketitle

\tableofcontents
\newpage

\section{Systemübersicht}
Die vorliegende Software simuliert einen \textbf{Digitalen Zwilling} eines industriellen Bohrsystems. Das Ziel dieser Anwendung ist es, die Brücke zwischen der klassischen Produktionstechnik und modernen KI-Methoden wie dem \textbf{Condition Monitoring} und der \textbf{Predictive Maintenance} (vorausschauende Wartung) greifbar zu machen.

Der digitale Zwilling nutzt physikalische Echtzeitdaten, um den unsichtbaren Zustand ("State") des Werkzeugs zu schätzen und das Risiko eines plötzlichen Werkzeugbruchs vorherzusagen.

\section{Physikalische Sensordaten}
Bevor die KI eine Entscheidung trifft, werden physikalische Basisdaten generiert. Diese folgen etablierten Formeln des Maschinenbaus:

\begin{itemize}
    \item \textbf{Drehmoment ($M_d$):} Die mechanische Last wird über die Kienzle-Formel berechnet. Sie hängt maßgeblich vom spezifischen Schnittkraftwert $k_{c1.1}$ des Werkstoffs und dem gewählten Vorschub $f$ ab. Einheit: \textbf{Nm}.
    \item \textbf{Schwinggeschwindigkeit ($v_{rms}$):} Die Vibration wird als Effektivwert der Schwinggeschwindigkeit simuliert. Einheit: \textbf{mm/s}. Ein Anstieg korreliert direkt mit zunehmendem Verschleiß.
    \item \textbf{Temperatur ($T$):} Die thermische Last an der Schneidkante in \textbf{°C}. Sie steigt progressiv mit der Schnittgeschwindigkeit $v_c$ und dem Reibungskoeffizienten des verschlissenen Werkzeugs.
\end{itemize}

\section{Mathematik der Prognose: Berechnung der TTF}
Die \textbf{Time To Failure (TTF)} ist der zentrale Kennwert der vorausschauenden Wartung. Sie gibt an, wie viele Zyklen (Bohrungen) das Werkzeug unter den aktuellen Bedingungen noch sicher absolvieren kann.



Die Berechnung erfolgt in vier Schritten:
\begin{enumerate}
    \item \textbf{Datenerfassung:} Das System speichert eine Historie der Verschleißwerte ($w$) über die Zeitachse der Zyklen ($c$).
    \item \textbf{Trendanalyse:} Mittels \textbf{linearer Regression} wird die Verschleißrate $m$ ermittelt.
    \item \textbf{Regressionsformel:} Die Steigung $m$ wird nach der Methode der kleinsten Quadrate berechnet:
    \begin{equation}
        m = \frac{n\sum(c \cdot w) - \sum c \sum w}{n\sum c^2 - (\sum c)^2}
    \end{equation}
    \item \textbf{Projektion:} Die TTF ist die Distanz zur kritischen 100\%-Marke, dividiert durch die aktuelle Verschleißrate:
    \begin{equation}
        TTF = \frac{100 - w_{aktuell}}{m}
    \end{equation}
\end{enumerate}

\section{Anwendungsszenarien für die Praxis}

\subsection{Szenario A: Werkstoff-Wechsel (Reaktionsgeschwindigkeit)}
\textbf{Auftrag:} Stellen Sie \textit{Baustahl (S235JR)} ein ($v_c = 160, f = 0,18$). Starten Sie die Simulation. Wechseln Sie bei Zyklus 50 im laufenden Betrieb auf \textit{Inconel (Superlegierung)}.
\begin{itemize}
    \item \textbf{Erwartung:} Das Drehmoment ($M_d$) verdoppelt sich fast augenblicklich.
    \item \textbf{Physik:} Die höhere Materialhärte fordert mehr Kraft pro mm² Spanquerschnitt.
    \item \textbf{KI:} Das System erkennt die neue Lastsituation und korrigiert das Bruchrisiko sofort nach oben.
\end{itemize}

\subsection{Szenario B: Grenzlasttest (Optimierung)}
\textbf{Auftrag:} Wählen Sie \textit{Titan-Legierung}. Setzen Sie $v_c$ auf 200 m/min und $f$ auf 0,40 mm/U. Deaktivieren Sie die \textbf{Kühlschmierung}.
\begin{itemize}
    \item \textbf{Erwartung:} Die Temperatur schießt über 700°C. Das Bruchrisiko steigt massiv an.
    \item \textbf{Physik:} Fehlende Kühlung führt zu thermischer Erweichung der Schneidkante.
    \item \textbf{KI:} Die KI kombiniert die Faktoren "Hohe Last" und "Keine Kühlung" und warnt vor einem Ausfall, noch bevor der Verschleiß 100\% erreicht.
\end{itemize}

\subsection{Szenario C: Adaptive Life-Extension}
\textbf{Auftrag:} Lassen Sie das Werkzeug bei Standardwerten verschleißen, bis die TTF unter 10 Zyklen sinkt. Reduzieren Sie nun $v_c$ und $f$ um jeweils 50\%.
\begin{itemize}
    \item \textbf{Erwartung:} Die TTF-Anzeige stabilisiert sich und steigt wieder an.
    \item \textbf{Physik:} Geringere Schnittkräfte reduzieren die mechanische Spannung im Werkzeug.
    \item \textbf{KI:} Die KI erkennt die Schonfahrt und berechnet eine neue, flachere Verschleißrate.
\end{itemize}

\section{Zusammenspiel von Physik und KI}
Die Wirksamkeit des Digitalen Zwillings entsteht durch die Symbiose zweier Denkweisen:

\subsection{Physik als Fundament (Kausalität)}
Die physikalischen Modelle sind \textbf{deterministisch}. Das bedeutet: Wenn wir Vorschub und Material kennen, können wir das Drehmoment exakt berechnen. Die Physik liefert die objektive Wahrheit über die Kräfte, die auf die Maschine wirken. Sie ist die Basis für jede Sensorik.

\subsection{KI als Interpretation (Situative Intelligenz)}
Die KI arbeitet \textbf{probabilistisch} (wahrscheinlichkeitsbasiert). Während die Physik sagt: "Es wirken 25 Nm", sagt die KI: "In Kombination mit dem hohen Alter des Bohrers bedeuten diese 25 Nm ein Bruchrisiko von 85\%". 
\begin{itemize}
    \item \textbf{Mustererkennung:} Die KI versteht Zusammenhänge (z.B. Temperaturanstieg + Vibration), die zu komplex für einfache Formeln sind.
    \item \textbf{Erfahrungswert:} Die KI agiert wie ein virtueller Experte, der Sensordaten bewertet und eine Handlungsempfehlung (Wartung!) ausspricht.
\end{itemize}



\subsection{Vergleichstabelle}
\begin{table}[h]
    \centering
    \renewcommand{\arraystretch}{1.5}
    \begin{tabular}{p{3.5cm} p{5.5cm} p{5.5cm}}
        \toprule
        \textbf{Aspekt} & \textbf{Physikalisches Modell} & \textbf{KI-Modell (Bayessch)} \\
        \midrule
        \textbf{Basis} & Formeln \& Naturgesetze & Daten \& Wahrscheinlichkeiten \\
        \textbf{Fokus} & "Warum passiert es?" (Kausal) & "Wie sicher ist es?" (Risiko) \\
        \textbf{Beispiel} & Berechnung von $M_d$ & Vorhersage der Restlaufzeit (TTF) \\
        \textbf{Stärke} & Absolute Präzision & Bewertung unklarer Situationen \\
        \bottomrule
    \end{tabular}
\end{table}

\section{Glossar}
\begin{description}
    \item[Inferenz:] Die Schlussfolgerung der KI aus vorliegenden Beweisen (Sensordaten).
    \item[Kienzle-Formel:] Standard-Formel zur Berechnung von Schnittkräften.
    \item[Condition Monitoring:] Die permanente Überwachung des Maschinenzustands.
\end{description}

\end{document}
