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

\definecolor{darkblue}{rgb}{0.0, 0.0, 0.5}
\definecolor{deepgreen}{rgb}{0.0, 0.5, 0.0}

\title{\textbf{Schulungsunterlage: KI-Zwilling Bohrsystem v21.8.2}}
\author{Industrie 4.0 Kompetenzzentrum | Condition Monitoring}
\date{19. Januar 2026}

\begin{document}

\maketitle

\section{Live-Analyse und Explainable AI (XAI)}
Die Software bietet im rechten Bereich ein Terminal, das die Entscheidungsfindung der KI transparent macht. Ein typischer Eintrag sieht wie folgt aus:

\begin{quote}
\texttt{[14:20:05] ZYK 680 (+10) | RISIKO: 75.4\% | Md: 18.2Nm | Vib: 4.22mm/s \\
➔ KI-LOGIK: [Alter: Alt | Last: HOCH | Temp: Normal | Kühlung: Aktiv]}
\end{quote}

Hierbei findet eine \textbf{Datenfusion} statt:
\begin{enumerate}
    \item \textbf{Messwerte:} Die kontinuierlichen Sensordaten (Nm, mm/s) werden angezeigt.
    \item \textbf{Interpretation:} Die KI übersetzt diese Werte mittels Schwellenwerten in Kategorien (z.B. $M_d > \theta \rightarrow$ Last: HOCH).
    \item \textbf{Inferenz:} Diese Kategorien triggern die entsprechende Zeile in der \textbf{CPT-Matrix} (Conditional Probability Table), um das Bruchrisiko zu berechnen.
\end{enumerate}



\section{Die korrigierte CPT-Logik (24 Zustände)}
Das System berechnet das Risiko basierend auf 24 möglichen Merkmalskombinationen (3 Altersstufen $\times$ 2 Lastzustände $\times$ 2 Temperaturzustände $\times$ 2 Kühlungszustände).

\begin{table}[h]
    \centering
    \begin{tabular}{ccc|ccc}
        \toprule
        \textbf{Last} & \textbf{Thermik} & \textbf{Kühlung} & \textbf{P(Kritisch) Neu} & \textbf{P(Kritisch) Alt} \\
        \midrule
        Normal & Normal & Aktiv & 1\% & 5\% \\
        HOCH & Normal & Aktiv & 10\% & 40\% \\
        HOCH & KRITISCH & AUS & 60\% & 95\% \\
        \bottomrule
    \end{tabular}
    \caption{Auszug aus der gewichteten Inferenz-Matrix.}
\end{table}

\end{document}
