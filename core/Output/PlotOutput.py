import string
from core.MPI_init import *
from .Output import Output
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import re

#=======================================================================================================================
class PlotOutput(Output):

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, response_data, input_params):
        self.tol = 1e-13
        self.max_qois_per_plot = 5
        self.response_data = response_data
        self.input_params = input_params
        self.get_QOI_prediction_type()
        self.get_plots()

    #-------------------------------------------------------------------------------------------------------------------
    def get_QOI_prediction_type(self):
        statistics_type = self.input_params['options']['statistics output']['statistics type']
        if statistics_type == 'robust':
            self.QOI_prediction_type = 'median'
        elif statistics_type == 'parametric':
            self.QOI_prediction_type = 'mean'
        elif statistics_type == 'mixed':
            self.QOI_prediction_type = 'median'
        else:
            raise ValueError("'options: statistics output: statistics type' is set incorrectly")

    #-------------------------------------------------------------------------------------------------------------------
    def get_discrete_colors(self, index):
        if index > 11:
            index = int(index - np.floor((index+1)/12)*12)
        colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99", "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928"]
        return colors[index]

    #-------------------------------------------------------------------------------------------------------------------
    def get_continuous_colors(self, jj, num_samples):
        array = np.ones((self.num_dims,1))
        discretization_color_non_norm = np.dot(self.response_data_x[jj],array)
        discretization_color_one_sample = (discretization_color_non_norm-np.amin(discretization_color_non_norm))/(np.amax(discretization_color_non_norm)-np.amin(discretization_color_non_norm))
        discretization_color = np.tile(discretization_color_one_sample,(1,num_samples)).flatten('C')
        return discretization_color
    
    #-------------------------------------------------------------------------------------------------------------------
    def get_markers(self, index):
        if index > 11:
            index = int(index - np.floor((index+1)/12)*12)
        marker_list = ["o", "^", "s", "v", "D", "<", "*", ">", "8", ".", "P", "X"]
        return marker_list[index]

    #-------------------------------------------------------------------------------------------------------------------
    def get_QOI_name(self, index):
        QOI_names = self.input_params['response data']['format']['QOI names']
        return QOI_names[index]

    #-------------------------------------------------------------------------------------------------------------------
    def get_selected_QOI_name(self, index):
        QOI_names = self.input_params['response data']['format']['QOI names']
        QOI_selection_index = np.array(self.input_params['response data']['selection']['QOI list'])-1
        selected_QOI_names = np.array(QOI_names)[QOI_selection_index]
        return selected_QOI_names[index]

    #-------------------------------------------------------------------------------------------------------------------
    def latexing_parameters(self, string):
        if "beta" in string:
            parameter = string.replace("beta", "$\\beta_{" ) + "}$"
        elif "gamma" in string:
            parameter = string.replace("gamma", "$\\gamma_{" ) + "}$"
        else:
            logging.info(f"Incorrect Latexing Parameter")
        return parameter

    #-------------------------------------------------------------------------------------------------------------------
    def get_model_fit_info(self):
        var_names_qoi = []
        index_gamma, index_beta = {}, {}
        self.df_model = pd.read_pickle( 'output/model_fit_statistics.pkl' )
        var_names_list = np.array( list(self.df_model.index.values) )[:,0]
        var_names = np.array( list(self.df_model.index.values) )[:,2]
        for ii in range(0,self.qoi_list.shape[0]):
            var_names_qoi.append(var_names[ii] + " " + var_names_list[ii])
        for ii in range(0,self.num_dims):
            index_gamma[ii] = np.where(var_names == "gamma"+str(ii+1))[0]
            index_beta[ii] = np.where(var_names == "beta"+str(ii+1))[0]
        self.index_gamma_model = index_gamma
        self.index_beta_model = index_beta

    #-------------------------------------------------------------------------------------------------------------------
    def get_summary_statistics_info(self):
        var_names_qoi = []
        self.df_summary = pd.read_pickle( 'output/summary_statistics.pkl' )
        self.qoi_list = self.input_params['response data']['selection']['QOI list']
        self.num_qoi = self.qoi_list.shape[0]
        var_names_list = np.array( list(self.df_summary.index.values) )[:,0]
        var_names = np.array( list(self.df_summary.index.values) )[:,1]
        for ii in range(0,var_names_list.shape[0]):
            var_names_qoi.append(self.latexing_parameters(var_names[ii]) + " " + var_names_list[ii])
        self.var_names_qoi_summary = np.array(var_names_qoi)
        self.index_gamma_summary = np.flatnonzero(np.core.defchararray.find(var_names,"gamma")!=-1)
        self.index_beta0_summary = np.where(var_names == "beta0")[0]
        self.exact_NaN = self.df_summary['exact'].isna().values.any()

    #-------------------------------------------------------------------------------------------------------------------
    def get_response_data_x(self):
        self.response_data_x = self.response_data.X
        self.num_dims = self.response_data_x[0].shape[1]

    #-------------------------------------------------------------------------------------------------------------------
    def get_response_data_y(self):
        self.response_data_y = self.response_data.Y

    #-------------------------------------------------------------------------------------------------------------------
    def exact_convergence_line(self, ii, x_streeq, error_streeq):
        gamma = self.df_summary['exact'].iloc[self.index_gamma_summary[ii]]
        x_min = np.amin(x_streeq)
        x_max = np.amax(x_streeq)
        y_min = np.amin(error_streeq)
        x = np.linspace(0.9*x_min, 1.1*x_max)
        bump_up_factor = 1.0
        c = y_min/(x_min**gamma)*bump_up_factor
        y = c*x**gamma
        return x, y

    #-------------------------------------------------------------------------------------------------------------------
    def streeq_convergence_line(self, index_beta0_summary_jj, index_gamma_model_ii_jj, index_beta_model_ii_jj, ii, jj):
        gamma = self.df_model[self.QOI_prediction_type].iloc[index_gamma_model_ii_jj]
        x_min = np.amin(self.response_data_x[jj][:,ii])
        x_max = np.amax(self.response_data_x[jj][:,ii])
        x = np.linspace(0.9*x_min, 1.1*x_max)
        beta_x = self.df_model[self.QOI_prediction_type].iloc[index_beta_model_ii_jj]
        if np.abs(float(self.df_summary['exact'].iloc[self.index_beta0_summary[jj]])) < self.tol:
            error_streeq = np.abs(beta_x*x**gamma)
        else:
            error_streeq = np.abs((beta_x*x**gamma)/self.df_summary['exact'].iloc[index_beta0_summary_jj])
        return x, error_streeq
    
    #-------------------------------------------------------------------------------------------------------------------
    def streeq_converged_value(self, index_beta0_summary_jj):
        beta_0_median = self.df_summary[self.QOI_prediction_type].iloc[index_beta0_summary_jj]
        beta_0_lower = self.df_summary['lower bound'].iloc[index_beta0_summary_jj]
        beta_0_upper = self.df_summary['upper bound'].iloc[index_beta0_summary_jj]
        return beta_0_median, beta_0_lower, beta_0_upper

    #-------------------------------------------------------------------------------------------------------------------
    def compute_error(self, jj):
        if np.abs(float(self.df_summary['exact'].iloc[self.index_beta0_summary[jj]])) < self.tol:
            error = np.abs(self.response_data_y[jj])
        else:
            error = np.abs((self.response_data_y[jj]-self.df_summary['exact'].iloc[self.index_beta0_summary[jj]])/self.df_summary['exact'].iloc[self.index_beta0_summary[jj]])
        return error

    #-------------------------------------------------------------------------------------------------------------------
    def compare_parameters_plot(self, index, parameter):  
         
        if parameter == "gamma":
            my_label = ["Median $\\gamma$", "95% Bootstrap CI", "Exact"]
        elif parameter == "beta":
            my_label = ["Median $\\beta_0$", "95% Bootstrap CI", "Exact"]
        if self.exact_NaN == True:
            my_label = my_label[:-1]
        for ii in range(0,index.shape[0]):
            xbound = np.array([[self.df_summary['lower bound'].iloc[index[ii]]], [self.df_summary['upper bound'].iloc[index[ii]]]])-self.df_summary[self.QOI_prediction_type].iloc[index[ii]]
            x=self.df_summary[self.QOI_prediction_type].iloc[index[ii]]
            y=self.var_names_qoi_summary[index[ii]]
            plt.errorbar(y=[self.var_names_qoi_summary[index[ii]]], x=self.df_summary[self.QOI_prediction_type].iloc[index[ii]], xerr=np.absolute(xbound), ecolor='k', capsize=10, color='k', label=my_label[1])
            plt.scatter(y=self.var_names_qoi_summary[index[ii]], x=self.df_summary[self.QOI_prediction_type].iloc[index[ii]], marker='o', facecolor='None', edgecolors='k', label=my_label[0])
            if self.exact_NaN == False:
                plt.scatter(y=self.var_names_qoi_summary[index[ii]], x=self.df_summary['exact'].iloc[index[ii]], marker='x', color='k', label=my_label[2])
                my_label = ["_nolegend_", "_nolegend_", "_nolegend_"]
            else:
                my_label = ["_nolegend_", "_nolegend_"]
        
        legend = plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3, borderaxespad=0, frameon=False)
        if parameter == "gamma":
            plt.xlabel(r'Observed Order of Accuracy')
            gamma_plot_path = os.path.join(Path.cwd(), 'plot', 'compare_parameters_gammas.pdf')
            plt.tight_layout()
            plt.savefig(gamma_plot_path, bbox_extra_artists=(legend,), bbox_inches='tight')
            plt.close()
            logging.info(f"    saved parameters plot to {gamma_plot_path}")
        elif parameter == "beta":
            plt.xlabel(r'QOI Value')       
            beta0_plot_path = os.path.join(Path.cwd(), 'plot', 'compare_parameters_beta0s.pdf')
            plt.tight_layout()
            plt.savefig( beta0_plot_path, bbox_extra_artists=(legend,), bbox_inches='tight' )
            plt.close()
            logging.info(f"    saved parameters plot to {beta0_plot_path}")

    #-------------------------------------------------------------------------------------------------------------------
    def order_of_accuracy_plot(self):   
        for dim in range(1, self.num_dims+1):
            num_plots = int(np.ceil(self.num_qoi/self.max_qois_per_plot))
            remainder = int(self.num_qoi - self.max_qois_per_plot*(num_plots-1))
            for plot_id in range(1, num_plots+1):
                if plot_id < num_plots:
                    self.order_of_accuracy_plot_subset(self.max_qois_per_plot, dim, plot_id, num_plots)
                else:
                    self.order_of_accuracy_plot_subset(remainder, dim, plot_id, num_plots)

    #-------------------------------------------------------------------------------------------------------------------
    def order_of_accuracy_plot_subset(self, num_qoi_per_plot, dim, plot_id, num_plots):
        for jj in range(0,num_qoi_per_plot):
            index = self.max_qois_per_plot*(plot_id - 1) + jj
            num_samples = len(self.response_data_y[index][0])
            x = np.transpose(np.tile(self.response_data_x[index][:,dim-1],(num_samples,1))).flatten('C')
            error = self.compute_error(index).flatten('C') 
            plt.scatter(x, error, marker=self.get_markers(index), c=self.get_continuous_colors(index, num_samples), cmap='viridis_r', edgecolors=self.get_discrete_colors(index))
            x_streeq, error_streeq = self.streeq_convergence_line(self.index_beta0_summary[index], self.index_gamma_model[dim-1][index], self.index_beta_model[dim-1][index], dim-1, index)
            plt.plot(x_streeq, error_streeq, color=self.get_discrete_colors(index), label=str(self.get_selected_QOI_name(index)), linewidth=1)

        x_exact, y_exact = self.exact_convergence_line(dim-1, x_streeq, error_streeq)
        plt.plot(x_exact, y_exact, 'k--', label="Exact", linewidth=1)
        
        cbar = plt.colorbar()
        cbar.set_label('Relative Refinement Scale')

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r'Dimension ' + str(dim) + ' Mesh Size')
        ylabel = plt.ylabel(r'Relative Error')
        plt.title('Order of Accuracy') 
        legend = plt.legend()
        plt.tight_layout()
        plot_file = self.order_of_accuracy_filepath(dim, plot_id, num_plots)
        plt.savefig( plot_file, bbox_extra_artists=(legend,ylabel), bbox_inches='tight' )
        plt.close()
        logging.info(f"    saved order of accuracy plot to {plot_file}")

    #-------------------------------------------------------------------------------------------------------------------
    def order_of_accuracy_filepath(self, dim, plot_id, num_plots):
        if num_plots == 1:
            filename = 'order_of_accuracy_dim_'+str(dim)+'.pdf'
        else:
            filename = 'order_of_accuracy_dim_'+str(dim)+'_part_'+str(plot_id)+'.pdf'    
        return os.path.join('plot', filename)

    def get_valid_filename(self, name):
        s = str(name).strip().replace(" ", "_")
        s = re.sub(r"(?u)[^-\w.]", "", s)
        if s in {"", ".", ".."}:
            raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)
        return s

    #-------------------------------------------------------------------------------------------------------------------
    def convergence_plot(self):   
        for ii in range(0,self.num_dims):
            for jj in range(0,self.num_qoi):  
                num_samples = len(self.response_data_y[jj][0])
                x = np.transpose(np.tile(self.response_data_x[jj][:,ii],(num_samples,1))).flatten('C')
                plt.scatter(x, self.response_data_y[jj], marker="o", c=self.get_continuous_colors(jj, num_samples), cmap='viridis_r', edgecolors='k', label="Discretization Data")
                cbar = plt.colorbar()  
                beta_0_median, beta_0_lower, beta_0_upper = self.streeq_converged_value(self.index_beta0_summary[jj])
                ybound = np.array([[beta_0_lower], [beta_0_upper]])-beta_0_median
                plt.errorbar(0, beta_0_median, yerr=np.absolute(ybound), ecolor='black', capsize=10, label="95% Bootstrap CI",  color='1')
                plt.scatter(0, beta_0_median, marker="o", facecolor="None", edgecolors='k', label="Median $\\beta_0$")
                cbar.set_label('Relative Refinement Scale')
                plt.xlabel(r'Dimension ' + str(ii+1) + ' Mesh Size')
                ylabel = plt.ylabel(str(self.get_selected_QOI_name(jj)))
                plt.title('Estimated Converged Value for '+str(self.get_selected_QOI_name(jj))) 
                legend = plt.legend()
                plt.tight_layout()
                filename = 'estimated_converged_value_dim_'+str(ii+1)+'_'+self.get_valid_filename(str(self.get_selected_QOI_name(jj)))+'.pdf'   
                plot_file = os.path.join(Path.cwd() / 'plot' / filename)
                plt.savefig( plot_file, bbox_extra_artists=(legend,ylabel), bbox_inches='tight' )
                plt.close()
                logging.info(f"    saved convergence plot to {plot_file}")

    #-------------------------------------------------------------------------------------------------------------------
    def get_plots(self): 
        self.get_summary_statistics_info()
        self.get_response_data_x()
        self.get_response_data_y()
        self.get_model_fit_info()
        self.compare_parameters_plot(self.index_gamma_summary, "gamma")
        self.compare_parameters_plot(self.index_beta0_summary, "beta")
        if self.exact_NaN == False:
            self.order_of_accuracy_plot()
        self.convergence_plot()

