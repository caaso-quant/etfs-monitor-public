import ipywidgets as widgets
import pandas as pd
# import bql
# bq = bql.Service()

class ETFFilter():
    def __init__(self, etf_fields):
        """Creates the dropdowns to filter the ETFs by a given fields list.

        :param etf_fields: List of tuples with all the fields from BQL of relevant \
            ETF information and a description.
        :type etf_fields: list
        """
        self.etf_fields = etf_fields
        self.callbacks = {
            'on_filter_click': None
        }

        self.selects = [self.create_select(field[0]) for field in etf_fields]

        filter_button = self.create_filter_button()
        clear_button = self.create_clear_button()

        accordion = widgets.Accordion(
            children=self.selects,
            selected_index=None,
        )
        for i, field in enumerate(etf_fields):
            accordion.set_title(i, field[1])
        
        container = widgets.Accordion(
            children=[widgets.VBox([clear_button, accordion, filter_button])],
            selected_index=None,
        )
        container.set_title(0, 'Filtrar ETFs')

        self.component = container

    def create_select(self, field):
        """Returns a multiple select widget with all the options \
            available of a given ETF field, and displays the amount \
                of ETFs with that option.

        :param field: BQL field name.
        :type field: string
        :return: The multiple select widget.
        :rtype: ipywidgets.SelectMultiple
        """		""""""
        # res = bq.execute("""
        #     get(COUNT(GROUP(ID, by=%s)).VALUE AS #COUNT) 
        #     for(filter(
        #         FUNDSUNIV(['active', 'primary']), 
        #         FUND_TYP=='ETF' 
        #         and EXCH_CODE=='BZ'
        #     ))
        # """%field)

        # df = res[0].df()
        # df.to_csv('./data/fields/%s.csv'%(field))
        df = pd.read_csv('./data/fields/%s.csv'%(field), index_col=0)

        options = [
            ("%s (%s)"%(index, df.loc[index, '#COUNT']), index) 
            for index in df.index.tolist()
        ]
        select = widgets.SelectMultiple(
            options=options,
            description='',
            disabled=False
        )
        
        return select

    def create_filter_button(self):
        """Creates a button that applies the selected filters.

        :return: The button widget.
        :rtype: ipywidgets.Button
        """
        filter_button = widgets.Button(
            description='Aplicar',
            disabled=False,
            button_style='info', 
            tooltip='Filtrar',
            icon='check' 
        )
        filter_button.layout.width = "100%"
        filter_button.layout.margin = "10px 0 0 0"

        def on_button_clicked(sender):
            filters = []
            for i, field in enumerate(self.etf_fields):
                filters.append({
                    "field":  field[0],
                    "value": self.selects[i].value
                })
            if self.callbacks['on_filter_click']:
                self.callbacks['on_filter_click'](filters)
            
        filter_button.on_click(on_button_clicked)

        return filter_button

    def create_clear_button(self):
        """Creates a button widget that clears all the selected \
            filter options.

        :return: The button widget.
        :rtype: ipywidgets.HBox
        """
        clear_button = widgets.Button(
            description='Limpar Filtros',
            disabled=False,
            button_style='warning', 
            tooltip='Filtrar',
            icon='trash'
        )
        clear_button.layout.margin = "0 0 10px 0"

        def on_clear_button_clicked(sender):
            for i in range(len(self.selects)):
                self.selects[i].value = []
        
        clear_button.on_click(on_clear_button_clicked)
        clear_button_container = widgets.HBox(
            children = [clear_button], 
            layout = widgets.Layout(display='flex', justify_content='flex-end'))

        return clear_button_container
        
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
