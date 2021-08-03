import ipywidgets as widgets
# import bql
# import bqviz as bqv
from bqplot import Figure, Pie, pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.efficient_frontier import EfficientFrontier

# bq = bql.Service()

class ETFViewer:
    def __init__(self, etf_fields, drivers):
        """Creates an empty component with the label requiring to select an ETF.
        """
        label = widgets.Label("Selecione um ETF.")
        self.component = widgets.VBox([label])
        self.component.layout.width = "40%"
        self.etf_fields = etf_fields
        self.drivers = drivers

    def set_etf(self, ids):
        """Set an ETF to the component and creates all tabs of visualizations.

        :param ids: List of BQL ids of the selected ETFs.
        :type ids: list
        """
        self.ids = ids
        self.component.children = [widgets.Label("Carregando...")]
        self.hist_returns = self.get_etf_historical_returns()

        out_error = widgets.Output()

        with out_error:
            titles = ['Performance', 'Volatilidade', 'Volume', 'Tracking', 'Holdings', 'Macro Factors', 'Infos']
            tab = widgets.Tab()
            tab.children = [
                self.create_performance_view(), 
                self.create_volatility_view(), 
                self.create_volume_view(), 
                self.create_delta_benchmark_view(),
                self.create_holdings_view(), 
                self.create_drivers_view(),
                self.create_infos_view(),
            ]
            for i, title in enumerate(titles):
                tab.set_title(i, title)
            tab.add_class('custom-tabs')
            
            self.component.children = [self.get_tabs_styles(), tab]
            return

        self.component.children = [out_error]
        
    def ids_as_string(self):
        return ','.join(["'%s'"%id for id in self.ids])
        
    def get_etf_historical_returns(self):
        """Gets a pandas DataFrame with the historical monthly returns of \
            the ETFs from a period of 3 years.

        :return: The monthly historical returns.
        :rtype: pd.DataFrame
        """        
        # bql_response = bq.execute(""" 
        #     get(PCT_DIFF(PX_LAST(M, dates=range(-3y, 0d)))) for([%s])
        # """%self.ids_as_string())
        
        # df = bql_response[0].df()
        # df.to_csv('./data/etfs/%s_returns.csv'%(self.ids[0]))
        df = pd.DataFrame()

        for id in self.ids:
            _df = pd.read_csv('./data/etfs/%s_returns.csv'%(id), index_col=0, parse_dates=['DATE'])
            _df.columns = ['DATE', 'RETURN']
            _df = _df.reset_index().pivot(index='DATE', columns='ID', values='RETURN')
            df = pd.concat([df, _df])

        return df
        
    def get_benchmark_return(self):
        """Gets a pandas DataFrame with the historical monthly return \
            of the ETFs benchmarks on a period of 3y.

        :return: The monthly historical returns.
        :rtype: pd.DataFrame
        """        
        # bql_response = bq.execute("get(FUND_BENCHMARK) for([%s])"%(self.ids_as_string()))
        # df = bql_response[0].df()
        # df.to_csv('./data/etfs/%s_benchmark.csv'%(self.ids[0]))
        df = pd.read_csv('./data/etfs/%s_benchmark.csv'%(self.ids[0]), index_col=0)
        df["FUND_BENCHMARK"] = df["FUND_BENCHMARK"] + " Index"
        self.benchmarks = df
        benchmarks = (df["FUND_BENCHMARK"]).tolist()
        # bql_response = bq.execute(""" 
        #     get(PCT_DIFF(PX_LAST(M, dates=range(-3y, 0d)))) for([%s])
        # """%",".join(["'%s'"%benchmark for benchmark in benchmarks]))
        # df = bql_response[0].df()
        # df.to_csv('./data/etfs/%s_returns.csv'%(benchmarks[0]))
        df = pd.read_csv('./data/etfs/%s_returns.csv'%(benchmarks[0]), index_col=0, parse_dates=['DATE'])
        df.columns = ['DATE', 'RETURN']
        df = df.reset_index().pivot(index='DATE', columns='ID', values='RETURN')
        return df

    def create_performance_view(self):
        """Creates a view that compares the returns of the ETF and of \
            the benchmark including a chart and a table.

        :return: A VBox container including the chart and the table.
        :rtype: ipywidgets.VBox
        """
        df2 = self.get_benchmark_return()
        self.benchmark_returns = df2
        df = self.hist_returns.merge(df2, left_index=True, right_index=True, how='inner')
        if df.shape[0] > 0:
            line_plot = self.create_line_plot(df, "Performance", "Date", "Return")
            performance_grid = self.create_performance_table(df)
            efficient_frontier = widgets.Label("Under construction")
            # efficient_frontier = self.create_efficient_frontier(df)
            return widgets.VBox([line_plot, performance_grid, efficient_frontier])
            # return widgets.VBox([widgets.Label("line"), performance_grid, efficient_frontier])
        else:
            return widgets.VBox([widgets.Label("Erro ao carregar.")])
       

    def create_performance_table(self, _df):
        """Creates the performance table comparing ETF and benchmark.

        :param _df: A pandas DataFrame containing the historical return \
            of the ETF and of the Benchmark.
        :type _df: pd.DataFrame
        :return: An output widget that displays the table.
        :rtype: ipywidgets.Output
        """
        df = _df.copy()       
        # print(df.index)
        df.index = df.index.strftime('%d/%m/%Y')
        df.index = df.index.rename("DATE")
        df.drop(df.iloc[0].name, inplace=True)
        make_float = lambda x: "{:,.2f}%".format(x)
        for i in range(len(df.columns)):
            df.iloc[:, i] = ['-' if j == None else "{:,.2f}%".format(j) for j in df.iloc[:, i]]
        fig = go.FigureWidget(data=[go.Table(
        header=dict(values=[df.index.name] + df.columns.tolist(),
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[df.index] + [df[col] for col in df.columns.tolist()], 
                    line_color='darkslategray',
                    fill_color='lightcyan',
                    align='left'))
        ])

        fig.update_layout(autosize=False, margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgb(0,0,0)', height=150)

        return fig
    
    def create_efficient_frontier(self, _df):
        """Creates the the efficient frontier chart with the selected ETFs.

        :param _df: A pandas DataFrame containing the historical return \
            of the ETFs.
        :type _df: pd.DataFrame
        :return: An output widget that displays the chart.
        :rtype: ipywidgets.Output
        """
        df = _df.copy()
        weights = [float(1/len(self.ids))]*len(self.ids)
        efficient_frontier = EfficientFrontier(df[self.ids], weights)
        
        if len(self.ids) < 2:
            return widgets.Label("Selecione mais de um ETF para visualizar a fronteira eficiente.")
        
        fig = efficient_frontier.trace_by_time()
        out = widgets.Output()
        with out:
            fig.show()

        return out
        

    def create_holdings_view(self):
        """Creates a view containing the holdings chart and a dropdown for \
            switching between ETFs.

        :return: A VBox container including the chart and the dropdown.
        :rtype: ipywidgets.VBox
        """
        dropdown = widgets.Dropdown(
            options=self.ids,
            description='',
            disabled=False
        )
        
        def on_dropdown_change(sender):
            id = dropdown.value
            chart_container.children = [widgets.Label("Loading...")]
            chart_container.children = [self.create_holdings_chart(id)]
            
        dropdown.observe(on_dropdown_change, names=['value'])
        
        chart_container = widgets.HBox([self.create_holdings_chart(self.ids[0])])
        
        return widgets.VBox([dropdown, chart_container])
    
    def create_holdings_chart(self, id):
        """Creates a view that shows a pizza chart and a table with \
            all the holdings and weights of the ETF.

        :return: A VBox container including the chart and the table.
        :rtype: ipywidgets.VBox
        """
        try:
            # bql_response = bq.execute("get(GROUPSORT(ID().weights)) for(HOLDINGS('%s'))"%id)
            # df = bql_response[0].df()
            # df.to_csv('./data/etfs/%s_holdings.csv'%(id))
            df = pd.read_csv('./data/etfs/%s_holdings.csv'%(id), index_col=0)
            df.index = df.index.rename('HOLDING')
            df.columns = ["WEIGHT"]
            make_float = lambda x: "{:,.2f}%".format(x)
            df2 = df.copy()
            df2.iloc[:, 0] = df2.iloc[:, 0].apply(make_float)
            
            fig = go.FigureWidget(data=[go.Table(
            header=dict(values=[df2.index.name] + df2.columns.tolist(),
                        line_color='darkslategray',
                        fill_color='lightskyblue',
                        align='left'),
            cells=dict(values=[df2.index] + [df2[col] for col in df2.columns.tolist()], 
                        line_color='darkslategray',
                        fill_color='lightcyan',
                        align='left'))
            ])

            fig.update_layout(autosize=False, margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgb(0,0,0)', height=300)

            df_big = df[df["WEIGHT"] > 2]
            df_low = df[df["WEIGHT"] < 2]

            soma = pd.DataFrame(df_low.head().sum(), columns=["WEIGHT"])
            soma.index = pd.Index(["Others"])
            df = pd.concat([df_big, soma])

            pie = Pie(sizes=df["WEIGHT"].tolist(), display_labels='outside', labels=df.index.tolist(), radius=120)
            fig2 = Figure(marks=[pie], animation_duration=1000, padding_x = 0, padding_y=0, fig_margin={"top":0, "bottom":0, "left": 0, "right":0})
            fig2.layout.height = "260px"
            fig2.layout.width = "85%"

            return widgets.VBox([fig2, fig])
        except:
            return widgets.Label("Erro ao carregar holdings.")
    
    def create_drivers_view(self):
        """Creates a view containing the drivers chart and a dropdown for \
            switching between ETFs.

        :return: A VBox container including the chart and the dropdown.
        :rtype: ipywidgets.VBox
        """
        dropdown = widgets.Dropdown(
            options=self.ids,
            description='',
            disabled=False
        )
        
        def on_dropdown_change(sender):
            id = dropdown.value
            chart_container.children = [widgets.Label("Loading...")]
            chart_container.children = [self.create_drivers_chart(id)]
            
        dropdown.observe(on_dropdown_change, names=['value'])
        
        chart_container = widgets.HBox([self.create_drivers_chart(self.ids[0])])
        
        return widgets.VBox([dropdown, chart_container])
    
    def create_drivers_chart(self, id):
        """Creates a view of a area-weighted chart of the absolute correlation \
            of the selected ETF and a list of drivers.

        :return: A VBox container with the chart.
        :rtype: ipywidgets.VBox
        """        
        # res = bq.execute("get(PCT_DIFF(PX_LAST(D, dates=range(-1y, 0d)))) for([%s])"
        #                  %(','.join(["'%s'"%driver for driver in self.drivers])))
        # drivers = res[0].df()
        # drivers.to_csv('./data/drivers/full_returns.csv')
        drivers = pd.read_csv('./data/drivers/full_returns.csv', index_col=0)
        drivers.columns = ["DATE", "RETURN"]
        drivers = drivers.reset_index().pivot(index="DATE", columns="ID", values="RETURN")
        join = self.hist_returns.merge(drivers, left_index=True, right_index=True)
        join = join.loc[:, [id] + self.drivers]
        corr = join.corr()
        fig = px.treemap(
            names = corr.iloc[:, 1:].columns.tolist(),
            parents = [corr.columns.tolist()[0] for x in range(len(corr.iloc[:, 1:].columns.tolist()))],
            values = corr.iloc[0, 1:].abs().tolist()
        )
        fig.update_traces(root_color="#333")
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25), paper_bgcolor='rgb(0,0,0)')
        
        container = go.FigureWidget(fig) 
        
        return widgets.VBox([container])

    def create_volatility_view(self):
        """Creates a view of a volatility chart with a window of\
            1 month from a period of 1 year.

        :return: A VBox container including a line chart.
        :rtype: ipywidgets.VBox
        """        
        # res = bq.execute("""
        #     get(std(group(
        #         px_last(dates=range(-1y, 0d)), 
        #         by=[
        #             YEAR(px_last(dates=range(-1y, 0d)).date),
        #             MONTH(px_last(dates=range(-1y, 0d)).date),
        #             ID
        #         ]
        #     )) AS #VOLATILITY) 
        #     for([%s])
        # """%self.ids_as_string())
        # df = res[0].df()
        # df.to_csv('./data/etfs/%s_vol.csv'%(self.ids[0]))
        df = pd.read_csv('./data/etfs/%s_vol.csv'%(self.ids[0]))
        df.drop(df.columns.difference(['DATE','ID.1','#VOLATILITY']), 1, inplace=True)
        df.columns = ["DATE", 'ID', "VOLATILITY"]
        df = df.pivot(index="DATE", columns="ID", values=["VOLATILITY"])
        df.columns = df.columns.droplevel(0)
        df.sort_index(inplace=True)
        # line_plot = bqv.LinePlot(df).set_style()
        # return widgets.VBox([line_plot.show()])
        return widgets.VBox([self.create_line_plot(df, "Volatility", "Date", "Vol")])

    def create_line_plot(self, df, title, xaxis, yaxis):
        plt.figure(title=title)

        colors = ["tab:blue", "tab:orange", "tab:purple", "tab:red", "tab:cyan"]

        for column in df.columns:
            plt.plot(df[column], label=column)

        plt.xlabel(xaxis)
        plt.ylabel(yaxis)

        out = widgets.Output()

        with out:
            plt.show()

        return out

    def create_infos_view(self):
        """Creates a view containing the infos list of the ETF and a dropdown for \
            switching between ETFs.

        :return: A VBox container including the list and the dropdown.
        :rtype: ipywidgets.VBox
        """
        dropdown = widgets.Dropdown(
            options=self.ids,
            description='',
            disabled=False
        )
        
        def on_dropdown_change(sender):
            id = dropdown.value
            chart_container.children = [widgets.Label("Loading...")]
            chart_container.children = [self.create_infos_list(id)]
            
        dropdown.observe(on_dropdown_change, names=['value'])
        
        chart_container = widgets.HBox([self.create_infos_list(self.ids[0])])
        
        return widgets.VBox([dropdown, chart_container])
    
    def create_infos_list(self, id):
        """Creates a view with a list of intrisinc characteristics of the ETF.

        :return: A VBox container with all the information.
        :rtype: ipywidgets.VBox
        """        
        bql_fields = ','.join([field[0] for field in self.etf_fields])
        display_fields = {
            "FUND_TOTAL_ASSETS": "Fund Total Assets",
            "EQY_SH_OUT": "EQY SH Out",
            "TRACKING_ERROR": "Tracking Error",
        }
        for field in self.etf_fields:
            display_fields[field[0]] = field[1]
        
        # res = bq.execute("""
        #     get(%s) 
        #     for('%s')
        # """%(','.join(display_fields.keys()), id))
        lines = []
        # for i in range(len(res)):
        for i in range(11):
            # df = res[i].df()
            # df.to_csv('./data/etfs/%s_%s.csv'%(id, df.columns[0]))
            try:
                df = pd.read_csv('./data/etfs/%s_%s.csv'%(id, [*display_fields][i]), index_col=0)
                if 'DATE' in df.columns:
                    df.drop('DATE', axis=1, inplace=True)
                if 'CURRENCY' in df.columns:
                    df.drop('CURRENCY', axis=1, inplace=True)
                column = df.columns[0]
                if column in display_fields:
                    display_column = display_fields[column]
                else:
                    display_column = column
                value = df[column].iloc[0]
                lines += [widgets.HTML("""
                    <div>
                        <span style='color: black'>%(column)s:</span> 
                        <span>%(value)s</span>
                    </div>
                """%{
                    "column": display_column,
                    "value": value
                })] 
            except:
                continue
        
        return widgets.VBox(lines)
        
    def create_volume_view(self):
        """Creates a view containg a line chart of the trades volume of the ETF.

        :return: A VBox container with the line chart.
        :rtype: ipywidgets.VBox
        """        
        # res = bq.execute("""
        #     get(NUM_TRADES(D, dates=range(-1y, 0d)) as #VOLUME) 
        #     for([%s])
        # """%self.ids_as_string())
        # df = res[0].df()
        # df.to_csv('./data/etfs/%s_volume.csv'%(self.ids[0]))
        df = pd.read_csv('./data/etfs/%s_volume.csv'%(self.ids[0]), index_col=0)
        
        df.drop(df.columns.difference(['DATE','#VOLUME']), 1, inplace=True)
        df.columns = ["DATE", "VOLUME"]
        df = df.reset_index().pivot(index="DATE", columns="ID", values=["VOLUME"])
        df.columns = df.columns.droplevel(0)
        df.sort_index(inplace=True)
        line_plot = self.create_line_plot(df, "Volume", "Date", "Volume")
        return widgets.VBox([line_plot])
    
    def create_delta_benchmark_table(self, delta):
        """Creates the delta table comparing ETF and benchmark.

        :param _df: A pandas DataFrame containing the delta of \
            the return of the ETF and of the Benchmark.
        :type _df: pd.DataFrame
        :return: An output widget that displays the table.
        :rtype: ipywidgets.Output
        """
        df = delta.mean()
        make_percentage = lambda x: "{:,.2f}%".format(x*100)
        make_float = lambda x: "{:,.2f}".format(x)
        df.iloc[:] = df.iloc[:].apply(make_float)
        above_avg = delta[delta > 0].count() / delta.count()
        above_avg = above_avg.apply(make_percentage)

        fig = go.FigureWidget(data=[go.Table(
        header=dict(values=["ETF", "Média", "% Over Performance"],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[df.index] + [df.tolist()] + [above_avg.tolist()], 
                    line_color='darkslategray',
                    fill_color='lightcyan',
                    align='left'))
        ])

        fig.update_layout(autosize=False, margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgb(0,0,0)', height=150)
        return fig

    def create_delta_benchmark_view(self):
        """Creates a view containg a line chart of the ETFs - benchmarks delta.

        :return: A VBox container with the line chart.
        :rtype: ipywidgets.VBox
        """        
        delta = self.hist_returns.copy()
        try:
            for column in delta.columns:
                delta[column] = delta[column] - self.benchmark_returns[self.benchmarks.loc[column, "FUND_BENCHMARK"]]
            line_plot = self.create_line_plot(delta, "Delta", "Date", "Delta")
            return widgets.VBox([line_plot, self.create_delta_benchmark_table(delta)])
        except:
            return widgets.VBox([widgets.Label("Erro ao carregar")])

    def get_tabs_styles(self):
        """Creates the styles of the tabs header.

        :return: An HTML widget with the styles.
        :rtype: ipywidgets.HTML
        """        
        html = """
            <style>
                .custom-tabs .lm-TabBar-tab{
                    backround-color: #202020 !important;
                }

                .custom-tabs .lm-TabBar-tab.lm-mod-current {
                    background-color: #DF6732 !important;
                }

                .custom-tabs .lm-TabBar-tab.lm-mod-current:before{
                    background: white !important;
                }
            </style>
        """
        html_widget = widgets.HTML(value=html)
        return html_widget

    def toggle_width(self):
        """Toggles the component width.
        """        
        if self.component.layout.width == "80%":
            self.component.layout.width = "40%"
        else:
            self.component.layout.width = "80%"

    def show(self):
        """Returns the component to be visualized
        """
        return self.component
        