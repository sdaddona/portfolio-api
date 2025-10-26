# portfolio_core.py
"""
Wrapper per esporre 'run_full_analysis' da portfolio_analysis.py
in modo che server.py e Render possano importarlo facilmente.
"""

import io
import os
import json
import traceback
import pandas as pd

# importa il tuo script principale
import portfolio_analysis as pa


def run_full_analysis(lots_text: str, bench: str = "VT", headless: bool = True):
    """
    Esegue l'analisi principale leggendo i lotti dal testo passato
    e utilizzando le funzioni già definite in portfolio_analysis.py.
    Ritorna un dizionario serializzabile in JSON per l'API web.
    """
    try:
        # Salva il testo su file temporaneo (come se fosse lots.txt)
        lots_path = "lots_web.txt"
        with open(lots_path, "w", encoding="utf-8") as f:
            f.write(lots_text)

        # Esegui il main del tuo script con gli stessi parametri
        # Simula l'analisi completa come se eseguissi: python portfolio_analysis.py --lots lots_web.txt --bench VT --alloc
        import subprocess
        cmd = [
            "python",
            "portfolio_analysis.py",
            "--lots", lots_path,
            "--bench", bench,
            "--alloc",
            "--headless" if headless else ""
        ]
        print("Eseguo comando:", " ".join([c for c in cmd if c]))
        subprocess.run([c for c in cmd if c], check=True)

        # Legge il file di output generato (es. summary.txt)
        out_summary = os.path.join("backtest_outputs", "02_summary.txt")
        summary_txt = ""
        if os.path.exists(out_summary):
            with open(out_summary, "r", encoding="utf-8") as f:
                summary_txt = f.read()

        result = {
            "status": "ok",
            "benchmark": bench,
            "summary_text": summary_txt,
            "note": "Analisi completata correttamente ✅"
        }

        return result

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "status": "error",
            "error": str(e),
            "traceback": tb
        }
