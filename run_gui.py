"""Entry point for MotorDesign2 — script and PyInstaller exe."""
import sys, os

if hasattr(sys, "_MEIPASS"):
    sys.path.insert(0, sys._MEIPASS)
else:
    _here = os.path.dirname(os.path.abspath(__file__))  # motordesign2/
    sys.path.insert(0, os.path.dirname(_here))           # parent
    sys.path.insert(0, _here)

from motordesign2.gui.app import main
main()
