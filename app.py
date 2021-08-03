import IPython.display
import ipywidgets as widgets
import pandas as pd

from components.etf_picker import ETFPicker
from components.etf_viewer import ETFViewer
from consts import *

class AppView():
	def show(self):
		"""Returns a widget display of the ETFs monitor APP
		"""
		etf_picker = ETFPicker(ETF_FIELDS, DRIVERS)
		etf_viewer = ETFViewer(ETF_FIELDS, DRIVERS)
		
		etf_picker.set_callback('on_width_change', etf_viewer.toggle_width)
		etf_picker.set_callback('on_select', etf_viewer.set_etf)
		
		hBox = widgets.HBox([etf_picker.show(), etf_viewer.show()])
		
		IPython.display.display(hBox)
                 
