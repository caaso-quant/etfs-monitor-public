import pandas as pd
from bqplot import market_map as mm
from bqplot import *
# import bqviz as bqv
import ipywidgets as widgets
# import bql
# bq = bql.Service()
import pdb

from components.etf_filter import ETFFilter

class ETFPicker:
    def __init__(self, etf_fields, drivers):
        # pdb.set_trace()
        """Gets information of all ETFs from BQL and creates all the widgets to filter ETFs.

        :param etf_fields: List of tuples with all the fields from BQL of relevant \
            ETF information and a description.
        :type etf_fields: list
        """
        self.etf_fields = etf_fields
        self.drivers = drivers
        df = self.get_df()
        self.unfiltered_df = df
        self.df = df.copy()
        self.callbacks = {
            "on_select": None,
            "on_width_change": None
        }
        self.colorize_radio = None
        self.driver_select_dropdown = None

        self.group_by = self.etf_fields[0][0]

        self.loading = widgets.Label("Loading...")
        self.loading.layout.visibility = "hidden"
        
        btn_toggle = self.create_toggle_button()
        self.filter_container = self.create_filter_container()
        self.map_container = self.create_map_container()

        self.component = widgets.VBox([
            btn_toggle, self.filter_container, self.loading, self.map_container
        ])

        self.component.layout.width = "60%"

    def get_df(self):
        """Returns a pandas DataFrame with all ETFs listed in Brazil, with \
            1 week and 1 year returns, and additional fields.
        """
        fields_string = ','.join([field[0] for field in self.etf_fields])
        # res = bq.execute("""
        #     get(
        #         PCT_CHG(PX_LAST(MODE=CACHED, dates=range(-1W, 0D, frq=D), fill=PREV)) AS #RETURN_1w, 
        #         PCT_CHG(PX_LAST(MODE=CACHED, dates=range(-1Y, 0D, frq=D), fill=PREV)) AS #RETURN_1y,
        #         %s
        #     ) 
        #     for(filter(
        #         FUNDSUNIV(['active', 'primary']), 
        #         FUND_TYP=='ETF' 
        #         and EXCH_CODE=='BZ'
        #     ))
        # """%fields_string)
        # df = res[0].df()
        # df.to_csv('./data/brazil_etfs0.csv')
        df = pd.read_csv('./data/brazil_etfs0.csv', index_col=0)

        df.drop('DATE', axis=1, inplace=True)
        df.drop('CURRENCY', axis=1, inplace=True)
        # for i in range(1, len(res)):
        for i in range(1, 10):
            # df2 = res[i].df()
            # df2.to_csv('./data/brazil_etfs%s.csv'%str(i))
            df2 = pd.read_csv('./data/brazil_etfs%s.csv'%str(i), index_col=0)
            if 'DATE' in df2.columns:
                df2.drop('DATE', axis=1, inplace=True)
            if 'CURRENCY' in df2.columns:
                df2.drop('CURRENCY', axis=1, inplace=True)
            df = df.merge(df2, how='inner', left_index=True, right_index=True)
        return df.reset_index()

    def get_correlation_colors(self, driver):
        """Get the correlation values for all the ETFs and a given driver

        :param driver: BQL Driver ID
        :type driver: string
        :return: Pandas Series containing the correlation of the ETFs with the driver
        :rtype: pd.Series
        """
        etfs = self.df["ID"].tolist()
        join = [driver] + etfs
        join_string = ["'%s'"%x for x in join]

        # res = bq.execute("get(PCT_DIFF(PX_LAST(D, dates=range(-1y, 0d)))) for([%s])"
        #                  %(','.join(join_string)))
        # returns = res[0].df()
        # returns.to_csv('./data/drivers/%s_returns.csv'%(driver))
        returns = pd.read_csv('./data/drivers/%s_returns.csv'%(driver), index_col=0)
        returns.columns = ["DATE", "RETURN"]
        returns = returns.reset_index().pivot(index="DATE", columns="ID", values="RETURN")
        return returns.corr().loc[driver].drop(driver)
    
    def filter_df(self, filters):
        """Filters the original ETFs DataFrame by a list of filters.

        :param filters: List of filter fields and values selected.
        :type filters: list
        :return: Pandas DataFrame with the filtered ETFs
        :rtype: pd.DataFrame
        """
        df = self.unfiltered_df.copy()
        for filter in filters:
            if len(filter["value"]) > 0:
                df = df[df[filter["field"]].isin(filter["value"])]
        return df

    def create_map_container(self):
        """Creates a component of the MarketMap widget and interactive inputs.

        :return: VBox widget containing the components.
        :rtype: ipywidgets.VBox
        """
        driver_select_dropdown = self.create_driver_select_dropdown()
        map = self.create_map()
        options_container = widgets.HBox([
            self.create_colorize_radio_input(),
            self.create_group_by_dropdown()
        ])
        
        return widgets.VBox([driver_select_dropdown, map, options_container])
        
    def create_driver_select_dropdown(self):
        """Creates a widget of a dropdown to select a macroeconomic Driver \
            to colorize the MarketMap visualization by correlation.

        :return: A VBox container with the label and the widget.
        :rtype: ipywidgets.VBox
        """
        label_driver = widgets.Label("Driver:")
        dropdown = widgets.Dropdown(
            options=self.drivers,
            description='',
            disabled=False
        )
        
        def on_dropdown_change(sender):
            correlations = self.get_correlation_colors(dropdown.value)
            df = self.df.set_index("ID").merge(correlations, how='inner', left_index=True, right_index=True)
            df.reset_index(inplace=True)
            self.map.ref_data = df
            self.map.tooltip_fields = ["#RETURN_1y","#RETURN_1w", dropdown.value, self.group_by]
            self.map.color = df[dropdown.value].fillna(0).tolist()
            self.colorize_radio.value = "Correlação Driver"
            
        dropdown.observe(on_dropdown_change, names=['value'])
        
        self.driver_select_dropdown = dropdown
        return widgets.VBox([label_driver, dropdown])

    def create_toggle_button(self):
        """Creates a widget of a button to toggle the visualization \
            of the component.

        :return: The widget of the button.
        :rtype: ipywidgets.Button
        """
        btn_toggle = widgets.Button(description="Esconder")
        def btn_toggle_minus_click(sender):
            if self.callbacks['on_width_change'] is not None:
                self.callbacks['on_width_change']()
            self.component.layout.width = "20%"
            self.map_container.layout.visibility = 'hidden'
            self.filter_container.layout.visibility = 'hidden'
            btn_toggle._click_handlers.callbacks = []
            btn_toggle.on_click(btn_toggle_plus_click)
            btn_toggle.description = "Escolher ETF"
            
        self.btn_toggle_minus_click = btn_toggle_minus_click

        def btn_toggle_plus_click(sender):
            if self.callbacks['on_width_change'] is not None:
                self.callbacks['on_width_change']()
            self.component.layout.width = "60%"
            self.map_container.layout.visibility = None
            self.filter_container.layout.visibility = None
            btn_toggle._click_handlers.callbacks = []
            btn_toggle.on_click(btn_toggle_minus_click)
            btn_toggle.description = "Esconder"

        btn_toggle.on_click(btn_toggle_minus_click)
        btn_toggle.layout.margin = "10px 0px 10px 5px"
        return btn_toggle
    
    def update_map(self):
        map = self.create_map()
        container_children = list(self.map_container.children)
        container_children[1] = map
        self.map_container.children = container_children

    def create_filter_container(self):
        """Creates a component composed by dropdowns responsible for \
            filtering the ETFs.

        :return: The component to filter the ETFs.
        :rtype: ipywidgets.VBox
        """
        etf_filter = ETFFilter(self.etf_fields)
        
        def on_filter_click(filters):
            self.loading.layout.visibility = None
            self.map_container.layout.visibility = 'hidden'
            self.df = self.filter_df(filters)
            self.update_map()
            self.loading.layout.visibility = 'hidden'
            self.map_container.layout.visibility = None
            
        etf_filter.set_callback('on_filter_click', on_filter_click)

        return widgets.VBox([etf_filter.show()])
        
    def create_colorize_radio_input(self):
        """Creates a radio buttons input that changes the MarketMap \
            colorization metric. Can be: Driver Correlation, 1 year return, \
                1 week return.

        :return: The widget of the radio buttons.
        :rtype: ipywidgets.VBox
        """
        radio = widgets.RadioButtons(
            options=['Correlação Driver' , 'Return 1y', 'Return 1w'],
            description='',
            disabled=False
        )
        
        def on_radio_change(sender):
            if radio.value == 'Correlação Driver':
                correlations = self.get_correlation_colors(self.drivers[0])
                df = self.df.set_index("ID").merge(correlations, how='inner', left_index=True, right_index=True)
                df.reset_index(inplace=True)
                self.driver_select_dropdown.value = self.drivers[0]
                self.map.color = df[self.drivers[0]].fillna(0).tolist()
            elif radio.value == "Return 1y":
                self.map.color = self.df["#RETURN_1y"].fillna(0).tolist()
            elif radio.value == "Return 1w":
                self.map.color = self.df["#RETURN_1w"].fillna(0).tolist()
                
        radio.observe(on_radio_change, names=['value'])

        label_radio = widgets.Label("Colorir por:")
        label_radio.layout.margin = "10px 0px 0px 0px"

        self.colorize_radio = radio
        return widgets.VBox([label_radio, radio])

    def create_group_by_dropdown(self):
        """Creates a dropdown widget that changes the MarketMap group by option.

        :return: The dropdown widget.
        :rtype: ipywidgets.VBox
        """
        dropdown = widgets.Dropdown(
            options=[(field[1], field[0]) for field in self.etf_fields],
            description='',
            disabled=False
        )
        
        def on_dropdown_change(sender):
            self.group_by = dropdown.value
            self.update_map()
            #self.map.groups = self.df[self.group_by].fillna("").tolist()
            #self.map.tooltip_fields = ["#RETURN_1y", self.group_by]
            
        dropdown.observe(on_dropdown_change, names=['value'])
        
        label_select = widgets.Label("Agrupar por:")
        label_select.layout.margin = "10px 0px 0px 0px"

        self.group_by_dropdown = dropdown
        return widgets.VBox([label_select, dropdown])

    def create_map(self):
        """Creates the MarketMap of the ETFs with the restriction \
            of selecting only one ETF.

        :return: The MarketMap widget.
        :rtype: bqplot.market_map.MarketMap
        """
        col = ColorScale(scheme='RdYlGn')
        ax_c = ColorAxis(scale=col, label='teste', visible=False)
        if self.driver_select_dropdown is not None:
            driver = self.driver_select_dropdown.value
        else:
            driver = self.drivers[0]
        
        correlations = self.get_correlation_colors(driver)
        
        df = self.df.set_index("ID").merge(correlations, how='inner', left_index=True, right_index=True)
        df.reset_index(inplace=True)
        if self.colorize_radio is not None:
            self.colorize_radio.value = 'Correlação Driver'
        
        map = mm.MarketMap(
            names=df['ID'].tolist(), 
            ref_data=df,
            color= df[driver].fillna(0).tolist(), 
            scales={'color':col},
            axes=[ax_c],
            map_margin = {"top": 0, "bottom": 0, "left": 0, "right": 0},
            groups=df[self.group_by].fillna("").tolist(),
            tooltip_fields = ["#RETURN_1y","#RETURN_1w", driver, self.group_by]
        )

        def onchange(sender):
            # map.selected = [map.selected[-1]]
            if self.callbacks['on_select'] is not None:
                self.btn_toggle_minus_click(sender)
                self.callbacks['on_select'](map.selected)

        map.observe(onchange, 'selected')
        self.map = map
        return map

    def set_callback(self, method, func):
        """Sets a callback to the component.

        :param method: The method for the callback to be setted.
        :type method: string
        :param func: The function of the callback.
        :type func: function
        """
        self.callbacks[method] = func 
        
    def show(self):
        """Returns the component to be visualized
        """
        return self.component
