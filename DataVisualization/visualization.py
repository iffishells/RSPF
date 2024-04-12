import os

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt
class DataVisualization:

    def __init__(self):
        self.data = 'sold_date'
        self.target_variable = 'price'

        self.parent_path_for_saving_html_visualization = os.path.join('..','Datasets','Visualization','html')
        self.parent_path_for_saving_image_visualization = os.path.join('..','Datasets','Visualization','image')
        
        # self.parent_path_for_saving_bar_plot_of_eac_country = 'Datasets/Visualization/bar_plot_of_each_country'
        # self.parent_path_for_saving_preprocessed_grouped_data = 'Datasets/preprocessed_data/grouped_data'

        os.makedirs(self.parent_path_for_saving_html_visualization,exist_ok=True)
        os.makedirs(self.parent_path_for_saving_image_visualization,exist_ok=True)
        # os.makedirs(self.parent_path_for_saving_preprocessed_grouped_data, exist_ok=True)
        # os.makedirs(self.parent_path_for_saving_bar_plot_of_eac_country,exist_ok=True)

        pass

    def Visualize(self, data: pd.DataFrame,
                  title=None):

        # Create a Plotly figure
        fig = go.Figure()

        # Add a trace to the figure
        fig.add_trace(go.Scatter(x=data[self.data], y=data[self.target_variable], name=title))

        # Update trace information and layout
        fig.update_traces(hoverinfo='text+name', mode='lines+markers')
        fig.update_layout(legend=dict(y=0.1, traceorder='reversed', font_size=12))
        fig.update_layout(title=str(title),width=1400,height=600)

        # Enable range selector for the x-axis
        fig.update_xaxes(rangeslider_visible=True,
                         rangeselector=dict(buttons=list([dict(step="all")])))
        # Update y-axis label
        fig.update_yaxes(title=f'{self.target_variable}')

        pio.write_image(fig,
                       os.path.join(self.parent_path_for_saving_image_visualization,title),
                       format='png')

        # pio.write_html(fig,
        #                os.path.join(self.parent_path_for_saving_html_visualization,title),
        #                format='html'
        #                )

        fig.write_html(f'{os.path.join(self.parent_path_for_saving_html_visualization)}/{title}.html')

        
        # fig.write_image('.png'))
        

    def __call__(self,
                 df=pd.DataFrame,
                 scatter_plot=False,
                 plot_down_sampled_data=False
                 ):
        if scatter_plot:

            saving_grouped_data_dir = os.path.join('..','Datasets','preprocessed_data')
            os.makedirs(saving_grouped_data_dir ,exist_ok=True)
            for name , group_df in df.groupby(['state','city']):
                if group_df.shape[0]>10000:
                    filterted_df = group_df[group_df['bed']<=4]
                    filterted_df.sort_values(by='sold_date',inplace=True)
                    filterted_df = filterted_df[filterted_df['sold_date'].notna()]
                    # Drop duplicate rows
                    filterted_df = filterted_df.drop_duplicates()

                    data_name = '_'.join(list(name))
                    print(data_name)

                    print('Size : ', group_df.shape)
                    filterted_df.to_csv(f'{saving_grouped_data_dir}/{data_name}.csv',index=False)

                    self.Visualize(data=filterted_df, title=data_name)


        if plot_down_sampled_data == True:

            saving_grouped_data_dir = os.path.join('..','Datasets','preprocessed_data','Weekly_downsampled')
            os.makedirs(saving_grouped_data_dir ,exist_ok=True)
            for name , group_df in df.groupby(['state','city']):
                if group_df.shape[0]>10000:
                    filterted_df = group_df[group_df['bed']<=4]
                    filterted_df.sort_values(by='sold_date',inplace=True)
                    filterted_df = filterted_df[filterted_df['sold_date'].notna()]
                    # Drop duplicate rows
                    filterted_df = filterted_df.drop_duplicates()

                    data_name = '_'.join(list(name))


                    # Assuming 'filtered_df' is your DataFrame with columns 'sold_date' and 'price'
                    sample = filterted_df[['sold_date','price']]
                    
                    # Convert 'sold_date' column to datetime
                    sample['sold_date'] = pd.to_datetime(sample['sold_date'])
                    
                    # Set 'sold_date' column as index
                    sample.set_index('sold_date', inplace=True)
                    
                    # Resample the data into monthly intervals and calculate the mean price within each month
                    downsampled_data = sample.resample('W').mean()
                    
                    downsampled_data.reset_index(inplace=True)
                    
                    print('Size : ', downsampled_data.shape)
                    downsampled_data.to_csv(f'{saving_grouped_data_dir}/weekly_downsampled_{data_name}.csv',index=False)

                    self.Visualize(data=downsampled_data, title=f'weekly_data_{data_name}')
















