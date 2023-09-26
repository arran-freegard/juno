#from rootalias import *
import ROOT
from ROOT import *
import math
from math import sin, cos, pi, exp
import cmath
import numpy as np
from npy_append_array import NpyAppendArray
from pprint import pprint
import iminuit
from iminuit import Minuit, cost
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
import time
from time import sleep
#import root_numpy
import pickle
import json
import pandas as pd
import csv
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt
from scipy.stats import skewnorm
import random
import os
import glob
from array import array
import sys
from scipy import interpolate
from datetime import datetime
FIGURE_DIR = 'figures/'

def get_graph_from_hist(h1):
    xs = []
    ys = []
    x_errs = []
    y_errs = []

    for i in range(1, h1.GetNbinsX() + 1):
        xs.append(h1.GetBinCenter(i))
        ys.append(h1.GetBinContent(i))
        x_errs.append(h1.GetBinWidth(i) / 2.)
        y_errs.append((h1.GetBinContent(i))**0.5)

    return TGraphErrors(len(xs), np.array(xs), np.array(ys), np.array(x_errs), np.array(y_errs))
def set_graph_style(gr, **kwargs):
    label_title_size = kwargs.get('label_title_size', 28)

    gr.SetMarkerStyle(24)
    gr.SetMarkerSize(1.2)
    gr.SetMarkerColor(kBlack)

    axiss = [gr.GetXaxis(), gr.GetYaxis()]
    if type(gr) == TGraph2D:
        axiss = [gr.GetXaxis(), gr.GetYaxis(), gr.GetZaxis()]

    for axis in axiss:
        axis.SetTitleOffset(1.2)
        axis.CenterTitle()
        axis.SetTitleFont(43)
        axis.SetLabelFont(43)
        axis.SetLabelSize(label_title_size)
        axis.SetTitleSize(label_title_size)
    # gr.GetYaxis().SetDecimals() # same label width
    # gr.GetYaxis().SetMaxDigits(3) # changes power of 10
    gr.SetLineWidth(2)
    gr.SetTitle('')
def set_h1_style(h1, **kwargs):
    label_title_size = kwargs.get('label_title_size', 28)

    h1.GetYaxis().SetTitleOffset(1.3)
    h1.GetXaxis().SetTitleOffset(1.2)
    h1.GetXaxis().CenterTitle()
    h1.GetYaxis().CenterTitle()
    h1.GetXaxis().SetTitleFont(43)
    h1.GetYaxis().SetTitleFont(43)
    h1.GetXaxis().SetLabelFont(43)
    h1.GetYaxis().SetLabelFont(43)
    h1.GetXaxis().SetLabelSize(label_title_size)
    h1.GetYaxis().SetLabelSize(label_title_size)
    h1.GetXaxis().SetTitleSize(label_title_size)
    h1.GetYaxis().SetTitleSize(label_title_size)
    h1.GetYaxis().SetNdivisions(510, 1)
    h1.GetXaxis().SetNdivisions(510, 1)
    h1.SetLineWidth(2)
    h1.SetTitle('')
  class NuOscFcn_ME:
    deg = pi / 180.
    
    def __init__(self, is_normal_ordering, baseline):
        self.is_normal_ordering = is_normal_ordering
        self.baseline = baseline     #km #52.75
    def __call__(self, x, par):
        energy = x[0]             # MeV
        th12 = par[0] * self.deg  # rad
        th13 = par[1] * self.deg  # rad
        dm2_s = par[2]            # eV^2, small mass splitting
        dm2_l = par[3]            # eV^2, large mass splitting

        dm221 = dm2_s
        if self.is_normal_ordering:
            dm232 = dm2_l
            dm231 = dm232 + dm2_s
        else:
            dm232 = -dm2_l# -0.093*1.e-3  
            dm231 = dm232 + dm2_s


        energy /= 1000.         # GeV
            
        rho = 2.45
        
        Ye = 0.5
        
        a = -1.52 * 10**(-4) *Ye * rho * energy
            
        dm2ee = cos(th12)**2 * dm231 + sin(th12)**2 * dm232
        
        s212_t =  sin(th12)**2 * (1+2*cos(th12)**2 * ((cos(th13)**2 * a)/dm221) + 3 * cos(th12)**2 * cos(2*th12)*((cos(th13)**2 * a)/dm221)**2 )
        
        dm221_t = dm221 * (1- cos(2 * th12) * ((cos(th13)**2 * a)/dm221) + 2 * sin(th12)**2 * cos(th12)**2 * ((cos(th13)**2 * a)/dm221)**2)
        
        s213_t = sin(th13)**2 * (1 + 2*cos(th13)**2 * (a/dm2ee))
        
        dm231_t = dm231 * (1 - (a/dm231) * ((cos(th12)**2 * cos(th13)**2 - sin(th13)**2) - sin(th12)**2 * cos(th12)**2 * cos(th13)**2 * (cos(th13)**2 * a/dm221)))
     
        dm232_t = dm232 * (1 - (a/dm232) * ((sin(th12)**2 * cos(th13)**2 - sin(th13)**2) + sin(th12)**2 * cos(th12)**2 * cos(th13)**2 * (cos(th13)**2 * a/dm221)))
        
        c212_t = 1 - s212_t
        c213_t = 1 - s213_t
        
        s2212_t = 4 * c212_t * s212_t   #sin^2(2 θ12) = 4 cos^2(θ12) sin^2(θ12)
        s2213_t = 4 * c213_t * s213_t   #sin^2(2 θ13) = 4 cos^2(θ13) sin^2(θ13)

        conv = 1000/(4*197.3269804)   #Conversion from eV2*km/GeV #rpp2022-rev-phys-constants.pdf
        #conv = 1.266932679815373
        #d21 = conv * dm221 * self.baseline / (energy)
        #d31 = conv * dm231 * self.baseline / (energy)
        #d32 = conv * dm232 * self.baseline / (energy)
        
        d21_t = conv * dm221_t * self.baseline / (energy)
        d31_t = conv * dm231_t * self.baseline / (energy)
        d32_t = conv * dm232_t * self.baseline / (energy)
        
        
        
        #prob = 1.
        #prob += - cos(th13)**4 * sin(2*th12)**2 * sin( d21 )**2
        #prob += - sin(2*th13)**2 * (cos(th12)**2 * sin( d31 )**2 + sin(th12)**2 * sin( d32 )**2)
  
        prob = 1.
        prob += - c213_t**2 * s2212_t * sin( d21_t )**2
        prob += - s2213_t * (c212_t * sin( d31_t )**2 + s212_t * sin( d32_t )**2)

        return prob

    class IbdCrossSection:
    mass_e = 0.511               # MeV
    mass_n = 939.565             # MeV
    mass_p = 938.272             # MeV
    mass_delta = mass_n - mass_p # MeV  #difference between neutron mass and proton mass

    def __init__(self):
        # print('mass_delta = {}'.format(self.mass_delta))
        # print('self.mass_delta + self.mass_e = {}'.format(self.mass_delta + self.mass_e))
        pass

    def get_xsec(self, energy_nu):
        energy_e = energy_nu - self.mass_delta                #electron/positron energy = energy of neutrino - mass difference between neutron and proton
        momentum_e = (energy_e**2 - self.mass_e**2)**0.5          #positron momentum = sqrt(E^2-M^2)  E^2 = M^2 + P^2
        sigma = 0.0952 * (energy_e * momentum_e) # e-42 cm^2    #Cross section = ...* energy * momentum

        return sigma

    class ibdCrossSection_theta:
    m_p = 938.27208816  #proton mass
    m_e = 0.51099895    #electron mass
    m_n = 939.56542052  #neutron mass
    delta_m = m_n - m_p
    y2 = (delta_m**2 - m_e**2)/2
    f = 1#
    g = 1.2701#1.26#
    f_2 = 3.706
    sigma_0 = 0.0952/((f**2 + 3 * g**2)) ##*10**(-42))
   
    def E_0(self, E_nu):
        E_0 = E_nu - self.delta_m
        return E_0
    
    def p_0(self, E_nu):
        p_0 = math.sqrt(self.E_0(E_nu)**2 - self.m_e**2)
        return p_0
    
    def v_0(self, E_nu):
        v_0 = self.p_0(E_nu)/self.E_0(E_nu)
        return v_0

    def E_1(self, E_nu, costheta):
        E_1 = self.E_0(E_nu) * (1 - E_nu * (1 - self.v_0(E_nu) * costheta)/self.m_p) - self.y2/self.m_p
        return E_1
    def p_1(self, E_nu, costheta):
        p_1 = math.sqrt(self.E_1(E_nu, costheta)**2-self.m_e**2)
        return p_1
    def v_1(self, E_nu, costheta):
        v_1 = self.p_1(E_nu, costheta)/self.E_1(E_nu, costheta)
        return v_1
    
    def Gamma(self, E_nu, costheta):
        gamma = 2*(self.f+self.f_2) * self.g * ((2 * self.E_0(E_nu) + self.delta_m) * (1 - self.v_0(E_nu) * costheta)-self.m_e**2/self.E_0(E_nu))+(self.f**2 + self.g**2) * (self.delta_m * (1 + self.v_0(E_nu) * costheta) + self.m_e**2/self.E_0(E_nu))+(self.f**2 + 3 * self.g**2) * ((self.E_0(E_nu) + self.delta_m)*(1 - 1/self.v_0(E_nu) * costheta) - self.delta_m)+ (self.f**2 - self.g**2) * ((self.E_0(E_nu) + self.delta_m) * (1 - 1/self.v_0(E_nu) * costheta) - self.delta_m) * self.v_0(E_nu) * costheta
        return gamma    
    
    def dsigdcos(self, E_nu, costheta):
        sigma = self.sigma_0/2 * ((self.f**2 + 3 * self.g**2) + (self.f**2 - self.g**2) * self.v_1(E_nu, costheta) * costheta ) * self.E_1(E_nu, costheta) * self.p_1(E_nu, costheta) - self.sigma_0 * self.Gamma(E_nu, costheta) * self.E_0(E_nu) * self.p_0(E_nu)/(2 * self.m_p)
        return sigma
    
    
    def XSection(self, aEnu):
        tsigma = 0;
        CalNUM = int(1e4);
        denC = 2. / CalNUM;
        for i in range(0,CalNUM):
            xc = 0;
            xc = -1. + i * denC;
            tsigma += self.dsigdcos(aEnu, xc) * denC;
  
        return tsigma;
tf_common = TFile('data/JUNOInputs2022_01_06.root')                   
h_ibd = tf_common.Get('IBDXsec_VogelBeacom_DYB')
#h_ibd = tf_common.Get('IBDXsec_StrumiaVissani')
h_ibd.SetDirectory(0)
h_ibd.Scale(10**(42))# cm^2


enus=[]
xsecs=[]
for i in range(0, h_ibd.GetNbinsX()):
        enus.append(h_ibd.GetBinCenter(i))
        xsecs.append(h_ibd.GetBinContent(i))

Xsec_gr = TGraph(len(enus), np.array(enus), np.array(xsecs))
def fluxes_CI():
    tf_common = TFile('data/JUNOInputs2022_01_06.root')                   
    h_flux = tf_common.Get('HuberMuellerFlux_DYBbumpFixed')
    h_flux.SetDirectory(0)
    enu_bins=[]
    fluxes=[]
    for i in range(0,h_flux.GetNbinsX()):
        enu_bins.append((h_flux.GetBinCenter(i)))
        fluxes.append(h_flux.GetBinContent(i))
    h_flux_New = np.interp(np.arange(1.804 + 0.01, 13, 0.01), enu_bins, fluxes)
    t_fluxes = TGraph(len(enu_bins), np.array(enu_bins), np.array(fluxes))
    #return h_flux_New
    return t_fluxes
def fluxes():
    tf_common = TFile('data/JUNOInputs2022_01_06.root')                   
    h_fluxU235 = tf_common.Get('HuberMuellerFlux_U235')
    h_fluxU235.SetDirectory(0)
    h_fluxU238 = tf_common.Get('HuberMuellerFlux_U238')
    h_fluxU238.SetDirectory(0)
    h_fluxPu239 = tf_common.Get('HuberMuellerFlux_Pu239')   
    h_fluxPu239.SetDirectory(0)
    h_fluxPu241 = tf_common.Get('HuberMuellerFlux_Pu241')
    h_fluxPu241.SetDirectory(0)

    enu_bins=[]
    fluxU235_event_count=[]
    fluxU238_event_count=[]
    fluxPu239_event_count=[]
    fluxPu241_event_count=[]


    for i in range(1,h_fluxU235.GetNbinsX()+1):
        enu_bins.append(h_fluxU235.GetBinCenter(i))
        fluxU235_event_count.append(h_fluxU235.GetBinContent(i))
        fluxU238_event_count.append(h_fluxU238.GetBinContent(i))
        fluxPu239_event_count.append(h_fluxPu239.GetBinContent(i))
        fluxPu241_event_count.append(h_fluxPu241.GetBinContent(i))

    t_fluxU235 = TGraph(len(enu_bins), np.array(enu_bins), np.array(fluxU235_event_count))
    t_fluxU238 = TGraph(len(enu_bins), np.array(enu_bins), np.array(fluxU238_event_count))
    t_fluxPu239 = TGraph(len(enu_bins), np.array(enu_bins), np.array(fluxPu239_event_count))
    t_fluxPu241 = TGraph(len(enu_bins), np.array(enu_bins), np.array(fluxPu241_event_count))

    return t_fluxU235, t_fluxU238, t_fluxPu239, t_fluxPu241
  class EnergyResFcn:

    def __init__(self):
        pass

    def __call__(self, x, par):
        e_vis = x[0]   # MeV
        a = par[0]     # poisson
        b = par[1]     # Cherenkov, non-uniformity
        c = par[2]     # positron annihilation, dark current

        sigma_over_e = (a**2 / e_vis + b**2 + c**2 / e_vis**2)**0.5
#        sigma_over_e = (a**2  + b**2/ e_vis + c**2 / e_vis**2)**0.5

        return sigma_over_e
def fluxbump():

    tf_common = TFile('data/JUNOInputs2022_01_06.root')                   
    h_DYBFluxBump_ratio = tf_common.Get('DYBFluxBump_ratio')
    h_DYBFluxBump_ratio.SetDirectory(0)
    h_DYBFluxBump_ratio.Rebin(4)
    
    e_nus = []
    DYBFluxBump_event_counts = []
    
    for nu_energy in range(1, h_DYBFluxBump_ratio.GetNbinsX() + 1):
        e_nu = h_DYBFluxBump_ratio.GetBinCenter(nu_energy)
        DYBFluxBump_event_count = h_DYBFluxBump_ratio.GetBinContent(nu_energy)
        e_nus.append(e_nu)
        DYBFluxBump_event_counts.append(DYBFluxBump_event_count)
    gr_DYBFluxBump = TGraph(len(e_nus), np.array(e_nus), np.array(DYBFluxBump_event_counts))

    return gr_DYBFluxBump
def SNF_N_NonEq():

    tf_common = TFile('data/JUNOInputs2022_01_06.root')                   
    h_SNF_ratio = tf_common.Get('SNF_FluxRatio')
    h_SNF_ratio.SetDirectory(0)
    h_NonEq_ratio = tf_common.Get('NonEq_FluxRatio')
    h_NonEq_ratio.SetDirectory(0)
    
    h_SNF = tf_common.Get('SNF_FluxRatio')#SNF_VisibleSpectrum')
    h_SNF.SetDirectory(0)
    h_NonEq = tf_common.Get('NonEq_FluxRatio')#NonEq_VisibleSpectrum')
    h_NonEq.SetDirectory(0)
    
    e_nus = []
    SNF_event_counts = []
    NonEq_event_counts = []

    for nu_energy in range(1, h_SNF.GetNbinsX() + 1):
        e_nu = h_SNF.GetBinCenter(nu_energy)
        SNF_event_count = h_SNF.GetBinContent(nu_energy)
        NonEq_event_count = h_NonEq.GetBinContent(nu_energy)
        e_nus.append(e_nu)
        SNF_event_counts.append(SNF_event_count)
        NonEq_event_counts.append(NonEq_event_count)
    gr_SNF = TGraph(len(e_nus), np.array(e_nus), np.array(SNF_event_counts))
    gr_NonEq = TGraph(len(e_nus), np.array(e_nus), np.array(NonEq_event_counts))

    return gr_SNF, gr_NonEq
def get_unoscillated_spectrum(**kwargs):
    exposure = kwargs.get('exposure', 6)          # yr
    bin_width = kwargs.get('bin_width', 0.01)     # MeV
    baseline = kwargs.get('baseline', 52.5)       # km, e5 cm
    reactor_power = kwargs.get('reactor_power', 2.9) # 26.6   # GW, 2.9 + 2.9 + 2.9 + 2.9 + 2.9 + 2.9 + 4.6 + 4.6, page 4 of DocDB 7494

    
    target_mass = 20000000 #(20kton) (kg)
    hydrogen_frac = 0.1201
    H_abundance = 0.999885
    m_p_kg = 1.672621923*10**(-27)  #proton mass (kg)
    m_e_kg = 9.1093837015*10**(-31)  #electron mass (kg)
    
    #m_H = 1.0078250322
    
    t_fluxU235, t_fluxU238, t_fluxPu239, t_fluxPu241 = fluxes()
    
    proton_count = target_mass*hydrogen_frac*H_abundance/(m_p_kg+m_e_kg) * 1e-33
    
    print("proton count: ",proton_count)

    #proton_count = 1.43512                        # 1e33
    reactor_duty_cycle = 11. / 12.

    ReactorFlux_ci = fluxes_CI()

    ibd_xsec = ibdCrossSection_theta()
    
    sum_f_e = (0.58*202.36 + 0.07*205.99 + 0.30*211.12 + 0.05*214.26) *1.602176633E-3
    print(1/sum_f_e)
    #1 eV = 1.602E-19 J
    #1 GW = 10E9 J/s
    SNF, NonEq = SNF_N_NonEq()
    DYBFluxBump = fluxbump()
    e_nus=[]
    event_counts=[]
    i=0
    for nu_energy in np.arange(1.804 + 0.01, 13, 0.01):
        DYBFluxBump_count = DYBFluxBump.Eval(nu_energy)
        
        u235_ci = t_fluxU235.Eval(nu_energy)*0.58/sum_f_e   #/Energy per fission
        pu239_ci = t_fluxPu239.Eval(nu_energy)*0.30/sum_f_e
        u238_ci = t_fluxU238.Eval(nu_energy)*0.07/sum_f_e
        pu241_ci = t_fluxPu241.Eval(nu_energy)*0.05/sum_f_e
        
        # there are about 3.1536E7 seconds in a year
        #flux = ReactorFlux_ci.Eval(nu_energy)
        #flux = ReactorFlux_ci[i]
        #flux = flux * 3.1536 * 1/sum_f_e
        flux = (u235_ci + pu239_ci + u238_ci + pu241_ci)*3.1536
        xsec = Xsec_gr.Eval(nu_energy)
        #xsec = ibd_xsec.XSection(nu_energy)
        SNF_count = SNF.Eval(nu_energy)
        NonEq_count = NonEq.Eval(nu_energy)
        
        event_rate = flux * reactor_power * reactor_duty_cycle / (4 * math.pi * baseline**2) * xsec * proton_count * 1e7 # yr-1
        event_rate = event_rate*(1 + SNF_count + NonEq_count)
        event_rate = event_rate*DYBFluxBump_count
        event_count = event_rate * exposure # per MeV
        event_count *= bin_width
        e_nus.append(nu_energy)
        event_counts.append(event_count)
        i+=1
    gr = TGraph(len(e_nus), np.array(e_nus), np.array(event_counts))
    return gr
unoscillated_spectrum_YJ_1 = get_unoscillated_spectrum(bin_width=0.01, exposure=6, baseline=52.74, reactor_power=2.9)
unoscillated_spectrum_YJ_2 = get_unoscillated_spectrum(bin_width=0.01, exposure=6, baseline=52.82, reactor_power=2.9)
unoscillated_spectrum_YJ_3 = get_unoscillated_spectrum(bin_width=0.01, exposure=6, baseline=52.41, reactor_power=2.9)
unoscillated_spectrum_YJ_4 = get_unoscillated_spectrum(bin_width=0.01, exposure=6, baseline=52.49, reactor_power=2.9)
unoscillated_spectrum_YJ_5 = get_unoscillated_spectrum(bin_width=0.01, exposure=6, baseline=52.11, reactor_power=2.9)
unoscillated_spectrum_YJ_6 = get_unoscillated_spectrum(bin_width=0.01, exposure=6, baseline=52.19, reactor_power=2.9)
unoscillated_spectrum_TS_1 = get_unoscillated_spectrum(bin_width=0.01, exposure=6, baseline=52.77, reactor_power=4.6)
unoscillated_spectrum_TS_2 = get_unoscillated_spectrum(bin_width=0.01, exposure=6, baseline=52.64, reactor_power=4.6)
unoscillated_spectrum_DYB = get_unoscillated_spectrum(bin_width=0.01, exposure=6, baseline=215, reactor_power=17.4)
class energy_conversion():
    m_p = 938.27208816  #proton mass
    m_e = 0.51099895    #electron mass
    m_n = 939.56542052  #neutron mass
    
    delta = (m_n**2 - m_p**2 - m_e**2)/(2*m_p)
    
    def epsilon(self, Enu):
        epsilon = Enu/self.m_p
        return epsilon
    
    def kappa(self, Enu, costheta):
        kappa = (1 + self.epsilon(Enu))**2 - (self.epsilon(Enu)*costheta)**2
        return kappa
    
    def Ee(self,Enu,costheta):
        Ee = (((Enu-self.delta)*(1+self.epsilon(Enu))+self.epsilon(Enu)*costheta*math.sqrt((Enu-self.delta)**2-self.m_e**2*self.kappa(Enu,costheta)))/self.kappa(Enu,costheta))
        return Ee
    
    
    def Ee_int(self, aEnu):
        tsigma = 0;
        CalNUM = int(1e4);
        denC = 2. / CalNUM;
#pragma omp parallel for reduction(+ \: tsigma)
        for i in range(0,CalNUM):
            xc = 0;
            xc = -1. + i * denC;
            tsigma += self.Ee(aEnu, xc) * denC;
  
        return tsigma;

    def Ee_int_2(self, aEnu):
        energies=[]
        for i in np.arange(-1,1.1,0.1):
            theta = i
            energy = self.Ee(aEnu,theta)
            energies.append(energy)
        av_en = sum(energies)/len(energies)
        return av_en
en_con = energy_conversion()
ibd_xsec = ibdCrossSection_theta()
m_p = 938.27208816  #proton mass
enus=[]
Energy=[]
for nu_energy in np.arange(1.804 + 0.01, 13, 0.01):
    enu = nu_energy
    cth = -0.034*ibd_xsec.v_0(enu)+2.4*enu/ibd_xsec.m_p
    EE = en_con.Ee_int_2(enu)#*((1+nu_energy/m_p*(1-1/ibd_xsec.v_1(nu_energy,cth)*cth))/(1-en_con.Ee_int_2(enu)/m_p*(1-en_con.Ee_int_2(enu)*cth)))
    enus.append(enu)
    Energy.append(EE+0.511)

EnergyConvert_gr = TGraph(len(Energy), np.array(Energy), np.array(enus))
def apply_energy_conversion_matrix(h1):
    #(0.8->12) 0.0001
    print("**Energy Matrix**")
    Epmatrix = pd.read_csv("New_Full_EpromptMatrix_SV_08_12_0_001_-1_1_0_001.csv", header=None, index_col=False)
    Epmatrix.drop(Epmatrix.tail(1106).index, inplace = True)
    Epmatrix = Epmatrix.iloc[:,:-1106]
    EpmatrixNumpy = Epmatrix.to_numpy()
    h_vis_x = np.linspace(h1.GetXaxis().GetXmin(), h1.GetXaxis().GetXmax()-h1.GetXaxis().GetBinWidth(1), num=h1.GetNbinsX())
    h_vis_y = np.frombuffer(h1.GetArray(),'d', h1.GetNbinsX()+1, 0)
    print(len(h_vis_y))
    print(EpmatrixNumpy.shape[0])
    h_vis_convert = np.matmul(h_vis_y,EpmatrixNumpy)

    h_new = h1.Clone()
    h_new.SetDirectory(0)
    for i in range(1,h1.GetNbinsX()+1):
        h_new.SetBinContent(i,h_vis_convert[i])
    return h_new
def get_shape_unc():

    tf_common = TFile('data/JUNOInputs2022_01_06.root')                   
    h_DYBShapeUnc = tf_common.Get('DYBUncertainty')
    h_DYBShapeUnc.SetDirectory(0)
#    bin_width = 0.005

    enu_bins=[]
    DYBShapeUnc_event_count=[]
    
    for i in range(0,h_DYBShapeUnc.GetNbinsX()):
#        enu_bins.append(round(h_DYBShapeUnc.GetBinCenter(i)+(h_DYBShapeUnc.GetBinCenter(i+1)-h_DYBShapeUnc.GetBinCenter(i))/2,5))
        enu_bins.append(h_DYBShapeUnc.GetBinCenter(i))
        DYBShapeUnc_event_count.append(h_DYBShapeUnc.GetBinContent(i))

    New_DYBShapeUnc = np.interp(np.arange(1.804 + 0.01, 13, 0.01), enu_bins, DYBShapeUnc_event_count)
    
    e_nus = []
    DYBShapeUnc_event_counts = []
    i=0
    for nu_energy in np.arange(1.804 + 0.01, 13, 0.01):
        DYBShapeUnc_event_count = New_DYBShapeUnc[i]
        e_nus.append(nu_energy)
        DYBShapeUnc_event_counts.append(DYBShapeUnc_event_count)
        i+=1
    gr_DYBShapeUnc = TGraph(len(e_nus), np.array(e_nus), np.array(DYBShapeUnc_event_counts))

    #h_olddata = h_data.Clone()
    #data_DYBShapeUnc = h_data.Clone()
    
    #for e_nu in range(1, h_data.GetNbinsX() + 1):
    #    event_count = h_data.GetBinContent(e_nu)
    #    even_bin = h_data.GetBinCenter(e_nu)-0.02
    #    DYBShapeUnc_event_count = gr_DYBShapeUnc.Eval(even_bin)
#   #     print(event_count, SNF_event_count, NonEq_event_count)
    #    data_DYBFluxBump.SetBinContent(e_nu,event_count*(DYBFluxBump_event_count)) 
    #    h_data.SetBinContent(e_nu,event_count*(DYBFluxBump_event_count)) 
    
    

    return gr_DYBShapeUnc
def get_TAO_shape_unc():

    tf_common = TFile('data/JUNOInputs2022_01_06.root')                   
    h_TAOShapeUnc = tf_common.Get('TAOUncertainty')
    h_TAOShapeUnc.SetDirectory(0)
#    bin_width = 0.005

    enu_bins=[]
    TAOShapeUnc_event_count=[]
    
    for i in range(0,h_TAOShapeUnc.GetNbinsX()):
        enu_bins.append(round(h_TAOShapeUnc.GetBinCenter(i)+(h_TAOShapeUnc.GetBinCenter(i+1)-h_TAOShapeUnc.GetBinCenter(i))/2,5))
        TAOShapeUnc_event_count.append(h_TAOShapeUnc.GetBinContent(i))

    New_TAOShapeUnc = np.interp(np.arange(1.804 + 0.01, 13, 0.01), enu_bins, TAOShapeUnc_event_count)
    
    e_nus = []
    TAOShapeUnc_event_counts = []
    i=0
    for nu_energy in np.arange(1.804 + 0.01, 13, 0.01):
        TAOShapeUnc_event_count = New_TAOShapeUnc[i]
        e_nus.append(nu_energy)
        TAOShapeUnc_event_counts.append(TAOShapeUnc_event_count)
        i+=1
    gr_TAOShapeUnc = TGraph(len(e_nus), np.array(e_nus), np.array(TAOShapeUnc_event_counts))

    #h_olddata = h_data.Clone()
    #data_DYBShapeUnc = h_data.Clone()
    
    #for e_nu in range(1, h_data.GetNbinsX() + 1):
    #    event_count = h_data.GetBinContent(e_nu)
    #    even_bin = h_data.GetBinCenter(e_nu)-0.02
    #    DYBShapeUnc_event_count = gr_DYBShapeUnc.Eval(even_bin)
#   #     print(event_count, SNF_event_count, NonEq_event_count)
    #    data_DYBFluxBump.SetBinContent(e_nu,event_count*(DYBFluxBump_event_count)) 
    #    h_data.SetBinContent(e_nu,event_count*(DYBFluxBump_event_count)) 
    
    

    return gr_TAOShapeUnc
gr_DYBShapeUnc = get_shape_unc()
gr_TAOShapeUnc = get_TAO_shape_unc()
def evaluate_gaussian(mu, sigma, x):
    return math.exp(-(x - mu)**2 / (2 * sigma**2)) / (2 * math.pi * sigma**2)**0.5

def apply_leak_smearing_matrix(h1, h_matrix):

    # tf_matrix = TFile('data/make_reco_to_true_matrix.root')
    # h_matrix = tf_matrix.Get('h_matrix')
    # h_matrix.SetDirectory(0)
    # tf_matrix.Close()

    h_leakage = h1.Clone('h_leakage')
    for i in range(1, h_leakage.GetNbinsX() + 1):
        event_count = 0.
        for j in range(1, h1.GetNbinsX() + 1):
            probability = h_matrix.GetBinContent(j,i)
            event_count += (h1.GetBinContent(j) * probability)
        h_leakage.SetBinContent(i, event_count)
    
    
    c1 = TCanvas('c1', 'c1', 800, 600)
#    set_margin()
    gPad.SetLeftMargin(0.15)
    gPad.SetBottomMargin(0.15)
    #f1.Draw()
    h1.SetLineColor(kGreen)
    h1.Draw()
    h_leakage.SetLineColor(kRed)
    h_leakage.Draw('sames, hist')
    c1.Update()
    c1.SaveAs('{}/apply_leak_smearing.pdf'.format(FIGURE_DIR))
    #input('Press any key to continue.')
    return h_leakage

def apply_energy_resolution_smearing(h1, **kwargs):
    a = kwargs.get('a', 2.72 / 100)
    b = kwargs.get('b', 0.83 / 100)
    c = kwargs.get('c', 1.23 / 100)
    
    f_sigma_over_e = EnergyResFcn()
    f1 = TF1('f1', f_sigma_over_e, 1, 9, 3)
    f1.SetParameters(a, b, c)
    f1.SetParNames('a', 'b', 'c')
    f1.SetNpx(1000)

    bin_width = h1.GetBinWidth(1)
    h_smear = h1.Clone('h_smear')
    for i in range(1, h_smear.GetNbinsX() + 1):
        event_count = 0.
        for j in range(1, h1.GetNbinsX() + 1):
            e_vis = h1.GetBinCenter(j)
            sigma_vis = f1.Eval(e_vis) * e_vis
            probability = evaluate_gaussian(e_vis, sigma_vis, h_smear.GetBinCenter(i)) * bin_width
            event_count += h1.GetBinContent(j) * probability

        h_smear.SetBinContent(i, event_count)

        
    #c1 = TCanvas('c1', 'c1', 800, 600)
    #set_margin()
    # f1.Draw()
    #h1.SetLineColor(kGreen)
    #h1.Draw()
    #h_smear.SetLineColor(kRed)
    #h_smear.Draw('sames, hist')
    #c1.Update()
    #c1.SaveAs('{}/apply_energy_resolution_smearing.pdf'.format(FIGURE_DIR))
    #c1.Clear()
        
    
    return h_smear
def apply_lsnl_matrix(h_data):
    from scipy import interpolate
    print("*LSNL Matrix*")
    tf_common = TFile('data/JUNOInputs2022_05_08.root')                   
    h_MC_LSNL = tf_common.Get('J22rc0_positronScintNL')
    h_MC_LSNL.SetDirectory(0)

##Take x-axis and y-axis from LSNL root
    mc_enu = np.linspace(h_MC_LSNL.GetXaxis().GetXmin(), h_MC_LSNL.GetXaxis().GetXmax()-h_MC_LSNL.GetXaxis().GetBinWidth(1), num=h_MC_LSNL.GetNbinsX())
    mc_y = np.frombuffer(h_MC_LSNL.GetArray(),'d', h_MC_LSNL.GetNbinsX()+1, 0)
    mc_y = mc_y[1:]
#print(mc_y)
#print(h_MC_LSNL.GetBinContent(h_MC_LSNL.GetNbinsX()))

##Define Absolute Curve
    mc_lsnl = mc_enu*mc_y

##Define extrapolated curve from 0.8-12 (original is 1.02 - 12)
    mc_lsnl_f = interpolate.interp1d(mc_enu, mc_lsnl, fill_value='extrapolate', kind='linear')
#Define extended x and extended y
    mc_lsnl_ext_x = np.arange(0.8,12,h_MC_LSNL.GetXaxis().GetBinWidth(1))
    mc_lsnl_ext_y = mc_lsnl_f(mc_lsnl_ext_x)

#print(mc_lsnl_ext_x)
#print(mc_lsnl)

#print(mc_lsnl_ext_y)
##Define forward and backward interpolated curves
    f_MC_forward = interpolate.interp1d(mc_lsnl_ext_x, mc_lsnl_ext_y, kind='cubic')
    f_MC_backward = interpolate.interp1d(mc_lsnl_ext_y, mc_lsnl_ext_x, kind='cubic')

#Define even finer x-axis for interpolation
    x_original = np.arange(0.8,12,0.005)
    x_modified = f_MC_forward(x_original)
    x_backwards = f_MC_backward(x_original)

    edges_original = np.asarray(x_original)
    edges_modified = np.asarray(x_modified)
    edges_backward = np.asarray(x_backwards)

    #print(len(edges_original))
#print(edges_original)
#print(edges_modified)
#print(edges_backward)
    LSNLMatrix = _axisdistortion_python(edges_original,edges_modified,edges_backward)

    #h_data.Rebin(5)
    h_vis_x = np.linspace(h_data.GetXaxis().GetXmin(), h_data.GetXaxis().GetXmax()-h_data.GetXaxis().GetBinWidth(1), num=h_data.GetNbinsX())
    h_vis_y = np.frombuffer(h_data.GetArray(),'d', h_data.GetNbinsX()+1, 0)
    h_vis_y = h_vis_y[1:]
    h_vis_y_int = interpolate.interp1d(h_vis_x, h_vis_y, kind='cubic', fill_value='extrapolate')
    h_vis_y_new = h_vis_y_int(x_original)
    print(len(LSNLMatrix))
    print(len(h_vis_y_new))
    h_vis_convert = np.matmul(LSNLMatrix,h_vis_y_new[:-1])
    bin_width = 0.005
    energy_min = 0.8
    energy_max = 12
    #print(len(h_vis_convert))
    bin_count = int((energy_max - energy_min) / bin_width)
    #h_rec = h_data.Clone()
    #h_rec.SetDirectory(0)
    h_rec = TH1D('h_rec', 'h_rec', bin_count, energy_min, energy_max) # positron energy, kinetic + annihilationfor i in range(1,h_rec.GetNbinsX()+1):
    h_rec.SetDirectory(0)
    #print(h_rec.GetNbinsX()+1)
    print(len(h_vis_convert))
    print(len(range(1,h_rec.GetNbinsX()+1)))
    for i in range(1,h_rec.GetNbinsX()):
        h_rec.SetBinContent(i,h_vis_convert[i-1])
    h_rec.Scale(1/8)
    return h_rec
def _axisdistortion_python(edges_original,edges_modified,edges_backwards):

    edges_target = edges_original
    min_original = edges_original[0]
    min_target = edges_target[0]
    nbinsx = edges_original.size - 1
    nbinsy = edges_target.size - 1

    matrix = np.zeros(shape=(nbinsx, nbinsy))
    #matrix[:, :] = 0.0

    threshold = -1e10
    left_axis = 0
    right_axis = 0
    idxx0, idxx1, idxy = -1, -1, 0
    leftx_fine, lefty_fine = threshold, threshold
    while (
        leftx_fine <= threshold or leftx_fine < min_original or lefty_fine < min_target
    ):
        isx = edges_original[idxx0 + 1] < edges_backwards[idxx1 + 1]
        if isx:
            leftx_fine, lefty_fine = edges_original[0], edges_modified[0]
            left_axis = 0
            if (idxx0 := idxx0 + 1) >= nbinsx:
                return
        else:
            leftx_fine, lefty_fine = edges_backwards[0], edges_target[0]
            left_axis = 1
            if (idxx1 := idxx1 + 1) >= nbinsx:
                return

    width_coarse = edges_original[idxx0 + 1] - edges_original[idxx0]
    while True:
        right_orig = edges_original[idxx0 + 1]
        right_backwards = edges_backwards[idxx1 + 1]

        if right_orig < right_backwards:
            rightx_fine = right_orig
            righty_fine = edges_modified[idxx0 + 1]
            right_axis = 0
        else:
            rightx_fine = right_backwards
            righty_fine = edges_target[idxx1 + 1]
            right_axis = 1

        while lefty_fine >= edges_target[idxy + 1]:
            if (idxy := idxy + 1) > nbinsy:
                break

        ##
        ## Uncomment the following lines to see the debug output
        ## (you need to also uncomment all the `left_axis` lines)
        
        width_fine = rightx_fine-leftx_fine
        factor = width_fine/width_coarse
        #print(
        #        f"x:{leftx_fine:8.4f}→{rightx_fine:8.4f}="
        #        f"{width_fine:8.4f}/{width_coarse:8.4f}={factor:8.4g} "
        #        f"ax:{left_axis}→{right_axis} idxx:{idxx0: 4d},{idxx1: 4d} iidxy: {idxy: 4d} "
        #        f"y:{lefty_fine:8.4f}→{righty_fine:8.4f}"
        #)

        matrix[idxy, idxx0] = (rightx_fine - leftx_fine) / width_coarse

        if right_axis == 0:
            if (idxx0 := idxx0 + 1) >= nbinsx:
                break
            width_coarse = edges_original[idxx0 + 1] - edges_original[idxx0]
        elif (idxx1 := idxx1 + 1) >= nbinsx:
            break
        leftx_fine, lefty_fine = rightx_fine, righty_fine
        left_axis = right_axis

    return matrix
def get_fake_data(**kwargs):                         ##used by chi2 nmocostfunc class
    th12 = kwargs.get('th12', 33.65)        # deg
    th13 = kwargs.get('th13', 8.491)         # deg
    dm2_s = kwargs.get('dm2_s', 7.53)       # e-5 eV^2, small mass splitting
    dm2_l = kwargs.get('dm2_l', 2.453)       # e-3 eV^2, large mass splitting
    a_res = kwargs.get('a_res', 2.61 / 100)
    b_res = kwargs.get('b_res', 0.82 / 100)
    c_res = kwargs.get('c_res', 1.23 / 100)
    is_normal_ordering = kwargs.get('is_normal_ordering', True)
    h_smearing = kwargs.get('h_smearing', None)
    h_leakage = kwargs.get('h_leakage', None)
    res_abc = kwargs.get('res_abc', None) # [2.72 / 100, 0.83 / 100, 1.23 / 100]
    h_background = kwargs.get('h_background', None)
    bin_width = kwargs.get('bin_width', 0.001)
    signal_scaling = kwargs.get('signal_scaling', None)
    suffix = kwargs.get('suffix', '')
    selection_efficiency = kwargs.get('selection_efficiency', 0.822)
    exposure = kwargs.get('exposure', 6)
    alpha_NonEq = kwargs.get('alpha_NonEq', 0)
    alpha_SNF = kwargs.get('alpha_SNF', 0)
    dm2_s *= 1.e-5              # eV^2
    dm2_l *= 1.e-3              # eV^2

    bin_width = 0.001
    energy_min = 0.8
    energy_max = 12
    bin_count = int((energy_max - energy_min) / bin_width)
    h_vis = TH1D('h_vis', 'h_vis', bin_count, energy_min, energy_max) # positron energy, kinetic + annihilation
    selection_efficiency = 0.822
    ibd_xsec = ibdCrossSection_theta()
    m_p = 938.27208816  #proton mass
    m_e = 0.51099895    #electron mass
    m_n = 939.56542052  #neutron mass

    osc_prob_YJ_1 = NuOscFcn_ME(is_normal_ordering=is_normal_ordering, baseline = 52.74,) #2.9
    osc_prob_YJ_2 = NuOscFcn_ME(is_normal_ordering=is_normal_ordering, baseline = 52.82) #2.9
    osc_prob_YJ_3 = NuOscFcn_ME(is_normal_ordering=is_normal_ordering, baseline = 52.41) #2.9
    osc_prob_YJ_4 = NuOscFcn_ME(is_normal_ordering=is_normal_ordering, baseline = 52.49) #2.9
    osc_prob_YJ_5 = NuOscFcn_ME(is_normal_ordering=is_normal_ordering, baseline = 52.11) #2.9
    osc_prob_YJ_6 = NuOscFcn_ME(is_normal_ordering=is_normal_ordering, baseline = 52.19) #2.9
    osc_prob_TS_1 = NuOscFcn_ME(is_normal_ordering=is_normal_ordering, baseline = 52.77) #4.6
    osc_prob_TS_2 = NuOscFcn_ME(is_normal_ordering=is_normal_ordering, baseline = 52.64) #4.6
    osc_prob_DYB = NuOscFcn_ME(is_normal_ordering=is_normal_ordering, baseline = 215)    #17.4

    gr_unosc_YJ_1 = unoscillated_spectrum_YJ_1
    gr_unosc_YJ_2 = unoscillated_spectrum_YJ_2
    gr_unosc_YJ_3 = unoscillated_spectrum_YJ_3
    gr_unosc_YJ_4 = unoscillated_spectrum_YJ_4
    gr_unosc_YJ_5 = unoscillated_spectrum_YJ_5
    gr_unosc_YJ_6 = unoscillated_spectrum_YJ_6
    gr_unosc_TS_1 = unoscillated_spectrum_TS_1
    gr_unosc_TS_2 = unoscillated_spectrum_TS_2
    gr_unosc_DYB = unoscillated_spectrum_DYB    

    f1_YJ_1 = TF1('f1', osc_prob_YJ_1, 1.8, 9, 4)
    f1_YJ_1.SetParameters(th12, th13, dm2_s, dm2_l)
    f1_YJ_1.SetParNames('th12', 'th13', 'dm2_s', 'dm2_l')
    f1_YJ_1.SetNpx(1000)
    f1_YJ_2 = TF1('f1', osc_prob_YJ_2, 1.8, 9, 4)
    f1_YJ_2.SetParameters(th12, th13, dm2_s, dm2_l)
    f1_YJ_2.SetParNames('th12', 'th13', 'dm2_s', 'dm2_l')
    f1_YJ_2.SetNpx(1000)
    f1_YJ_3 = TF1('f1', osc_prob_YJ_3, 1.8, 9, 4)
    f1_YJ_3.SetParameters(th12, th13, dm2_s, dm2_l)
    f1_YJ_3.SetParNames('th12', 'th13', 'dm2_s', 'dm2_l')
    f1_YJ_3.SetNpx(1000)
    f1_YJ_4 = TF1('f1', osc_prob_YJ_4, 1.8, 9, 4)
    f1_YJ_4.SetParameters(th12, th13, dm2_s, dm2_l)
    f1_YJ_4.SetParNames('th12', 'th13', 'dm2_s', 'dm2_l')
    f1_YJ_4.SetNpx(1000)
    f1_YJ_5 = TF1('f1', osc_prob_YJ_5, 1.8, 9, 4)
    f1_YJ_5.SetParameters(th12, th13, dm2_s, dm2_l)
    f1_YJ_5.SetParNames('th12', 'th13', 'dm2_s', 'dm2_l')
    f1_YJ_5.SetNpx(1000)
    f1_YJ_6 = TF1('f1', osc_prob_YJ_6, 1.8, 9, 4)
    f1_YJ_6.SetParameters(th12, th13, dm2_s, dm2_l)
    f1_YJ_6.SetParNames('th12', 'th13', 'dm2_s', 'dm2_l')
    f1_YJ_6.SetNpx(1000)
    
    f1_TS_1 = TF1('f1', osc_prob_TS_1, 1.8, 9, 4)
    f1_TS_1.SetParameters(th12, th13, dm2_s, dm2_l)
    f1_TS_1.SetParNames('th12', 'th13', 'dm2_s', 'dm2_l')
    f1_TS_1.SetNpx(1000)
    
    f1_TS_2 = TF1('f1', osc_prob_TS_2, 1.8, 9, 4)
    f1_TS_2.SetParameters(th12, th13, dm2_s, dm2_l)
    f1_TS_2.SetParNames('th12', 'th13', 'dm2_s', 'dm2_l')
    f1_TS_2.SetNpx(1000)

    f1_DYB = TF1('f1', osc_prob_DYB, 1.8, 9, 4)
    f1_DYB.SetParameters(th12, th13, dm2_s, dm2_l)
    f1_DYB.SetParNames('th12', 'th13', 'dm2_s', 'dm2_l')
    f1_DYB.SetNpx(1000)
    
    for i in range(1, h_vis.GetNbinsX() + 1):
        e_vis = h_vis.GetBinCenter(i)   
        e_nu = e_vis
        #e_nu = EnergyConvert_gr.Eval(e_vis)
        event_count = gr_unosc_YJ_1.Eval(e_nu) * f1_YJ_1.Eval(e_nu) + gr_unosc_YJ_2.Eval(e_nu) * f1_YJ_2.Eval(e_nu) + gr_unosc_YJ_3.Eval(e_nu) * f1_YJ_3.Eval(e_nu) + gr_unosc_YJ_4.Eval(e_nu) * f1_YJ_4.Eval(e_nu) + gr_unosc_YJ_5.Eval(e_nu) * f1_YJ_5.Eval(e_nu) + gr_unosc_YJ_6.Eval(e_nu) * f1_YJ_6.Eval(e_nu) + gr_unosc_TS_1.Eval(e_nu) * f1_TS_1.Eval(e_nu)+gr_unosc_TS_2.Eval(e_nu) * f1_TS_2.Eval(e_nu) + gr_unosc_DYB.Eval(e_nu) * f1_DYB.Eval(e_nu)# + gr_unosc_HZ.Eval(e_nu) * f1_HZ.Eval(e_nu)
        event_count *= selection_efficiency  
        if event_count<0:
            event_count = 0
        h_vis.SetBinContent(i, event_count)

    h_vis = apply_energy_conversion_matrix(h_vis)
    h_rec = apply_lsnl_matrix(h_vis)
    h_rec = apply_energy_resolution_smearing(h_rec, a=0.02614, b=0.0064, c=0.012)

    h_rec_x = np.linspace(h_rec.GetXaxis().GetXmin(), h_rec.GetXaxis().GetXmax()-h_rec.GetXaxis().GetBinWidth(1), num=h_rec.GetNbinsX())
    h_rec_y = np.frombuffer(h_rec.GetArray(),'d', h_rec.GetNbinsX()+1, 0)
    h_rec_y = h_rec_y[1:]


    tf_Dubna = TFile('data/comparisons/spectra-dubna-normal-data.root')
    h_Dubna_rec = tf_Dubna.Get('juno_eres')
    h_Dubna_rec.SetDirectory(0)
    h_Dubna_x = np.linspace(h_Dubna_rec.GetXaxis().GetXmin(), h_Dubna_rec.GetXaxis().GetXmax()-h_Dubna_rec.GetXaxis().GetBinWidth(1), num=h_Dubna_rec.GetNbinsX())
    h_Dubna_y = np.frombuffer(h_Dubna_rec.GetArray(),'d', h_Dubna_rec.GetNbinsX()+1, 0)
    h_Dubna_y = h_Dubna_y[1:]
    plt.subplot(211)
    plt.plot(h_rec_x,h_rec_y, color="red")
    plt.plot(h_Dubna_x,h_Dubna_y, color="green", linestyle="dashed")
    
    plt.subplot(212)
    f_interp_new = interpolate.interp1d(h_rec_x, h_rec_y, kind='cubic', fill_value='extrapolate')
    h_int_new_y = f_interp_new(h_Dubna_x)
    plt.plot(h_Dubna_x,h_int_new_y/h_Dubna_y)
    plt.axhline(y=1)
    plt.ylim(0.95,1.05)
    plt.show()
    return h_rec

def load_background(Ri,Rf, Scale, exposure):
    day_count = 365.25 * exposure

    #set_margin()

    c1 = TCanvas('c1', 'c1', 800, 600)
    gPad.SetLeftMargin(0.15)
    gPad.SetBottomMargin(0.15) 
    print('#############')

    if Ri==0:
        tf_acc = TFile('data/det_ac_pdf_J22_0.70_12.0_1.90_2.50_17.2_1.50.root')
        h_acc = tf_acc.Get("e_prompt_0_200")
        h_acc.SetDirectory(0)
        h_acc.Rebin(2)
        h_acc.Reset()
        for i in np.arange(Ri,Rf,200):
            h_acctmp =  tf_acc.Get("e_prompt_"+str(i)+"_"+str(i+200))
            print(str(i)+"-"+str(i+200))
            h_acctmp.SetDirectory(0)
            h_acctmp.Rebin(2)
            h_acc.Add(h_acctmp)
        h_acctmp =  tf_acc.Get("e_prompt_"+str(5000)+"_"+str(5088))
        print(str(5000)+"-"+str(5088))
        h_acctmp.SetDirectory(0)
        h_acctmp.Rebin(2)
        h_acc.Add(h_acctmp)
    else:
        tf_acc = TFile('data/det_ac_pdf_J22_0.70_12.0_1.90_2.50_17.7_1.50.root')
        h_acc = tf_acc.Get("e_prompt_0_200")
        h_acc.SetDirectory(0)
        h_acc.Rebin(2)
        h_acc.Reset()
    
    if Ri == 5088:
        h_acctmp =  tf_acc.Get("e_prompt_"+str(5088)+"_"+str(5200))
        print(str(5088)+"-"+str(5200))
        h_acctmp.SetDirectory(0)
        h_acctmp.Rebin(2)
        h_acc.Add(h_acctmp)
    
    if Ri == 5200:
        h_acctmp =  tf_acc.Get("e_prompt_"+str(5200)+"_"+str(5400))
        print(str(5200)+"-"+str(5400))
        h_acctmp.SetDirectory(0)
        h_acctmp.Rebin(2)
        h_acc.Add(h_acctmp)
    
    if Ri == 5400:
        h_acctmp =  tf_acc.Get("e_prompt_"+str(5400)+"_"+str(5545))
        print(str(5400)+"-"+str(5545))
        h_acctmp.SetDirectory(0)
        h_acctmp.Rebin(2)
        h_acc.Add(h_acctmp)
    
        #h_acctmp = tf_common.Get('AccBkgHistogramAD')
        #h_acctmp.SetDirectory(0)
        #h_acc.Add(h_acctmp)
        #h_acc.Scale(day_count)


        
    tf_common = TFile('data/JUNOInputs2022_01_06.root')
                
        
    h_geo = tf_common.Get('GeoNuHistogramAD')
    h_li9 = tf_common.Get('Li9BkgHistogramAD')
    h_alpha_n = tf_common.Get('AlphaNBkgHistogramAD')
    h_fast_n = tf_common.Get('FnBkgHistogramAD')
    h_world_nu = tf_common.Get('OtherReactorSpectrum')
    h_world_nu_300 = tf_common.Get('OtherReactorSpectrum_L300km')
    h_world_nu.Add(h_world_nu_300)

    
    h_bgs = [h_geo, h_li9, h_alpha_n, h_fast_n, h_world_nu]
    for i, h_bg in enumerate(h_bgs):
        h_bg.Scale(Scale)
        h_bg.Scale(day_count)
        h_bg.SetDirectory(0)
    h_bgs.append(h_acc)
    return(h_bgs)
## For multiple spectra (len(data))
class NmoCostFunctionSimultaneousFit:
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, h_smearing,  h_leakages, res_abcs, h_datas, h_backgrounds_nom, signal_scalings, res_abcs_err, selection_efficiency, exposure, **kwargs):
        self.h_smearing = h_smearing
        self.h_leakages = h_leakages
        self.res_abcs = res_abcs
        self.h_datas = h_datas
        self.h_backgrounds_nom = h_backgrounds_nom
        self.signal_scalings = signal_scalings
        self.spectrum_count = len(h_datas)
        self.res_abcs_err = res_abcs_err
        self.selection_efficiency = selection_efficiency
        self.exposure = exposure

        #self.res_abcs = kwargs.get('res_abcs', [None for i in range(self.spectrum_count)])
        print('self.res_abcs = {}'.format(self.res_abcs))

    def __call__(self, th12, th13, dm2_s, dm2_l, a_res, b_res, c_res, 
                a_res_AV1,
                b_res_AV1,
                c_res_AV1,
        
                a_res_AV2,
                b_res_AV2,
                c_res_AV2,
        
                a_res_AV3,
                b_res_AV3,
                c_res_AV3,
                 
                eff_CV,
                eff_AV1,
                eff_AV2,
                eff_AV3,
                 
                geo_unc,
                li9_unc,
                alpha_unc,
                fast_unc,
                world_unc,
                acc_unc,
                
                alpha_NonEq,
                alpha_SNF,
                 
                alpha_CorRea,
                alpha_Det,
                 
                alpha_YJ,
                alpha_TS,
                alpha_DYB,
                alpha_HZ,
                               
                is_normal_ordering, **kwargs):
        print_chi2 = kwargs.get('print_chi2', False)
        
        w_YJ = 2.9*6/52.46
        w_TS = 4.6*2/52.71
        w_DYB = 17.4/215
        w_HZ = 17.4/265
        
        ws = [w_YJ,w_TS,w_DYB,w_HZ]
        
        wmin = min(ws) 
        wmax = max(ws)
        for i, w in enumerate(ws):
            ws[i] = (w-wmin) / (wmax-wmin)
        
#        h_background = scale_background(BG=self.h_backgrounds_nom[0],  geo_unc = geo_unc, li9_unc = li9_unc, alpha_unc = alpha_unc, fast_unc = fast_unc, world_unc = world_unc)
#        h_background_PV1 = scale_background(BG=self.h_backgrounds_nom[1], geo_unc = geo_unc, li9_unc = li9_unc, alpha_unc = alpha_unc, fast_unc = fast_unc, world_unc = world_unc)  #scale BG by volume
#        h_background_PV2 = scale_background(BG=self.h_backgrounds_nom[2], geo_unc = geo_unc, li9_unc = li9_unc, alpha_unc = alpha_unc, fast_unc = fast_unc, world_unc = world_unc)
#        h_background_PV3 = scale_background(BG=self.h_backgrounds_nom[3], geo_unc = geo_unc, li9_unc = li9_unc, alpha_unc = alpha_unc, fast_unc = fast_unc, world_unc = world_unc)
    
#        h_backgrounds = [
#        h_background.Clone(),
#        h_background_PV1.Clone(),
#        h_background_PV2.Clone(),
#        h_background_PV3.Clone()
#        ]
        h_FV = get_fake_data(
                th12=th12, th13=th13, dm2_s=dm2_s, dm2_l=dm2_l,
                a_res=a_res,
                b_res=b_res,
                c_res=c_res,
                is_normal_ordering=is_normal_ordering,
                h_smearing=self.h_smearing,
                h_leakage=None,
                res_abc=self.res_abcs[0],
                bin_width=self.h_datas[0].GetBinWidth(1),
                signal_scaling=self.signal_scalings[0],
                selection_efficiency = eff_CV,#self.selection_efficiency[0],
                exposure = self.exposure,
                alpha_NonEq = alpha_NonEq,
                alpha_SNF = alpha_SNF
            )
        h_PV1 = get_fake_data(
                th12=th12, th13=th13, dm2_s=dm2_s, dm2_l=dm2_l,
                a_res=a_res_AV1,
                b_res=b_res_AV1,
                c_res=c_res_AV1,
                is_normal_ordering=is_normal_ordering,
                h_smearing=self.h_smearing,
                h_leakage=self.h_leakages[1],
                res_abc=self.res_abcs[1],
                bin_width=self.h_datas[0].GetBinWidth(1),
                signal_scaling=self.signal_scalings[1],
                selection_efficiency = eff_AV1,  #self.selection_efficiency[1],
                exposure = self.exposure,
                alpha_NonEq = alpha_NonEq,
                alpha_SNF = alpha_SNF
            )
        h_PV2 = get_fake_data(
                th12=th12, th13=th13, dm2_s=dm2_s, dm2_l=dm2_l,
                a_res=a_res_AV2,
                b_res=b_res_AV2,
                c_res=c_res_AV2,
                is_normal_ordering=is_normal_ordering,
                h_smearing=self.h_smearing,
                h_leakage=self.h_leakages[2],
                res_abc=self.res_abcs[2],
                bin_width=self.h_datas[0].GetBinWidth(1),
                signal_scaling=self.signal_scalings[2],
                selection_efficiency = eff_AV2, #self.selection_efficiency[2],
                exposure = self.exposure,
                alpha_NonEq = alpha_NonEq,
                alpha_SNF = alpha_SNF
            )
        h_PV3 = get_fake_data(
                th12=th12, th13=th13, dm2_s=dm2_s, dm2_l=dm2_l,
                a_res=a_res_AV3,
                b_res=b_res_AV3,
                c_res=c_res_AV3,
                is_normal_ordering=is_normal_ordering,
                h_smearing=self.h_smearing,
                h_leakage=self.h_leakages[3],
                res_abc=self.res_abcs[3],
                bin_width=self.h_datas[0].GetBinWidth(1),
                signal_scaling=self.signal_scalings[3],
                selection_efficiency = eff_AV3, #self.selection_efficiency[3],
                exposure = self.exposure,
                alpha_NonEq = alpha_NonEq,
                alpha_SNF = alpha_SNF
            )
        
        #c1 = TCanvas('c1', 'c1', 800, 600)
        #set_margin()
        # f1.Draw()
        #h_FV.SetLineColor(kBlue)
        #h_FV.Draw('sames, hist')
        #h_PV1.SetLineColor(kRed)
        #h_PV1.Draw('sames, hist')
        #h_PV2.SetLineColor(kGreen)
        #h_PV2.Draw('sames, hist')
        #h_PV3.SetLineColor(kMagenta)
        #h_PV3.Draw('sames, hist')
        #c1.Update()
        #c1.SaveAs('{}/h_FV_PV123.pdf'.format(FIGURE_DIR))
        #c1.Clear()
        
        h_preds = [
            h_FV]#
            #h_PV1,
            #h_PV2,
            #h_PV3
        #]

        chi2s = []
        colors=["purple","purple","darkblue","darkred"]
        fig, ax1 = plt.subplots() 
        ax1.set_ylabel(r'$\Delta \chi^2$')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Event Count / 20 keV')
        for j, h_pred in enumerate(h_preds):
            chi2 = 0
            chi2_hist=[0]
            pred=[0]
            dat=[0]
            x_ = 0.8
            x_hist=[0.8]
            for i in range(1, h_pred.GetNbinsX() + 1):
                if h_pred.GetBinCenter(i) < 0.9 or h_pred.GetBinCenter(i) > 9:
                    continue
                    
    
                even_bin = h_pred.GetBinCenter(i)
                #TAOShapeUnc_event_count = gr_TAOShapeUnc.Eval(even_bin)
                #DYBShapeUnc_event_count = gr_DYBShapeUnc.Eval(even_bin)

                #print(even_bin, TAOShapeUnc_event_count)
                
                geo_BG = self.h_backgrounds_nom[j][0].GetBinContent(i)
                li9_BG = self.h_backgrounds_nom[j][1].GetBinContent(i) 
                alpha_BG = self.h_backgrounds_nom[j][2].GetBinContent(i)
                fast_BG = self.h_backgrounds_nom[j][3].GetBinContent(i) 
                world_BG = self.h_backgrounds_nom[j][4].GetBinContent(i) 
                acc_BG = self.h_backgrounds_nom[j][5].GetBinContent(i)
                
                ###(geo_BG*0.05)+(li9_BG*0.1)+(alpha_BG*0.5)+(fast_BG*0.2)+(world_BG*0.05)+(acc_BG*0)
                #print("Evaluate", (h_pred.GetBinContent(i) - self.h_datas[j].GetBinContent(i))**2 / (h_pred.GetBinContent(i)))
                chi2 += (h_pred.GetBinContent(i) - self.h_datas[j].GetBinContent(i))**2 / (h_pred.GetBinContent(i))
                
                #chi2 += (h_pred.GetBinContent(i)*(1 + alpha_CorRea + (alpha_YJ*ws[0]+alpha_TS*ws[1]+alpha_DYB*ws[2]+alpha_HZ*ws[3]) + alpha_Det) - self.h_datas[j].GetBinContent(i))**2 / (h_pred.GetBinContent(i))
                #chi2 += (h_pred.GetBinContent(i)*(1 + alpha_CorRea + (alpha_YJ*ws[0]+alpha_TS*ws[1]+alpha_DYB*ws[2]+alpha_HZ*ws[3]) + alpha_Det)+(geo_BG*(1+geo_unc)+li9_BG*(1+li9_unc)+alpha_BG*(1+alpha_unc)+fast_BG*(1+fast_unc)+world_BG*(1+world_unc)+acc_BG*(1+acc_unc)) - self.h_datas[j].GetBinContent(i))**2 / (h_pred.GetBinContent(i)+(geo_BG+li9_BG+alpha_BG+fast_BG+world_BG+acc_BG)+(h_pred.GetBinContent(i)*DYBShapeUnc_event_count)**2+((geo_BG*0.05)**2+(li9_BG*0.1)**2+(alpha_BG*0.5)**2+(fast_BG*0.2)**2+(world_BG*0.05)**2+(acc_BG*0)**2))
                chi2_hist.append(chi2)
                pred.append(h_pred.GetBinContent(i))
                dat.append(self.h_datas[j].GetBinContent(i))
                x_ += 0.02
                x_hist.append(x_)
            ax1.plot(x_hist,chi2_hist)
            ax2.plot(x_hist,pred,color="grey", linestyle="--")
            ax2.plot(x_hist,dat,color=colors[j], linestyle="--")
            ax2.set_xlabel(r"$E_{reco.}$ [MeV]")

            if j==0:
                plt.show()
                plt.clf()
                fig, ax1 = plt.subplots() 
                ax1.set_ylabel(r'$\Delta \chi^2$')
                ax2 = ax1.twinx()
                ax2.set_ylabel('Event Count / 20 keV')
                ax2.set_xlabel(r"$E_{reco.}$ [MeV]")
            chi2s.append(chi2)
            print("Sum "+str(j)+": "+str(sum(chi2s)))
        # https://arxiv.org/pdf/2111.03086.pdf
        plt.show()
        
        
        chi2s.append(((th12 - 33.4) / 0.77)**2)
        chi2s.append(((th13 - 8.6) / 0.12)**2)
        chi2s.append(((dm2_s - 7.4) / 0.21)**2)
        chi2s.append(((dm2_l - 2.5) / 0.028)**2)
        
   #     chi2s.append(((a_res - (2.681240 / 100) ) / ((2.681240 / 100)*0.01500648))**2)
   #     chi2s.append(((b_res - (0.7596468 / 100) ) / ((0.7596468 / 100)*0.007532366))**2)        
   #     chi2s.append(((c_res - (0.9982543 / 100) ) / ((0.9982543 / 100)*0.03469338))**2)
        
   #     chi2s.append(((a_res_AV1 - (self.res_abcs[1][0]) ) / ((self.res_abcs[1][0])*self.res_abcs_err[1][0]))**2)
   #     chi2s.append(((b_res_AV1 - (self.res_abcs[1][1]) ) / ((self.res_abcs[1][1])*self.res_abcs_err[1][1]))**2)        
   #     chi2s.append(((c_res_AV1 - (self.res_abcs[1][2]) ) / ((self.res_abcs[1][2])*self.res_abcs_err[1][2]))**2)
        
   #     chi2s.append(((a_res_AV2 - (self.res_abcs[2][0]) ) / ((self.res_abcs[2][0])*self.res_abcs_err[2][0]))**2)
   #     chi2s.append(((b_res_AV2 - (self.res_abcs[2][1]) ) / ((self.res_abcs[2][1])*self.res_abcs_err[2][1]))**2)        
   #     chi2s.append(((c_res_AV2 - (self.res_abcs[2][2]) ) / ((self.res_abcs[2][2])*self.res_abcs_err[2][2]))**2)
        
   #     chi2s.append(((a_res_AV3 - (self.res_abcs[3][0]) ) / ((self.res_abcs[3][0])*self.res_abcs_err[3][0]))**2)
   #     chi2s.append(((b_res_AV3 - (self.res_abcs[3][1]) ) / ((self.res_abcs[3][1])*self.res_abcs_err[3][1]))**2)        
   #     chi2s.append(((c_res_AV3 - (self.res_abcs[3][2]) ) / ((self.res_abcs[3][2])*self.res_abcs_err[3][2]))**2)
        
        
   #     chi2s.append(((eff_CV - (self.selection_efficiency[0]) ) / (0.00427))**2)
   #     chi2s.append(((eff_AV1 - (self.selection_efficiency[1]) ) / (0.00065))**2)        
   #     chi2s.append(((eff_AV2 - (self.selection_efficiency[2]) ) / (0.00087))**2)      
   #     chi2s.append(((eff_AV3 - (self.selection_efficiency[3]) ) / (0.00059))**2)
        
        
   #     chi2s.append(( (geo_unc) / (0.3))**2)   
   #     chi2s.append(( (li9_unc) / (0.20))**2)
   #     chi2s.append(( (alpha_unc) / (0.50))**2)
   #     chi2s.append(( (fast_unc) / (1.0))**2)
   #     chi2s.append(( (world_unc) / (0.02))**2)
   #     chi2s.append(( (acc_unc) / (0.01))**2)
        
   #     chi2s.append(( (alpha_NonEq) / (0.3))**2)  #Assumed 30% unc. (DocDB 7546 pg 5)
   #     chi2s.append(( (alpha_SNF) / (0.3))**2)   #Assumed 30% unc.

   #     chi2s.append(( (alpha_YJ/0.008))**2)
   #     chi2s.append(( (alpha_TS/0.008))**2)
   #     chi2s.append(( (alpha_DYB/0.008))**2)
   #     chi2s.append(( (alpha_HZ/0.008))**2)
        
   #     print("alpha_CorRea: "+str(alpha_CorRea))
   #     print("alpha_Det: "+str(alpha_Det))
        
   #     chi2s.append((alpha_CorRea / 0.02)**2)   #2%
   #     chi2s.append((alpha_Det/0.01)**2)        #1%
        
####abc syst####
  #      for i in range(0,2):
  #          for j in range(0,3):
  #              chi2s.append((self.res_abcs[i][j]/self.res_abcs_err[i][j])**2)
  #              print("penalty add abc: ",(self.res_abcs[i][j]/self.res_abcs_err[i][j])**2)
  #      
  #      chi2s.append((2.45/0.15)**2) ##ME
  #      
  #      chi2s.append( (1.2/(1.2*0.3))**2)    #Geo:
  #      chi2s.append((0.8/(0.8*0.01))**2)    #Accidental
  #      chi2s.append((0.8/(0.8*0.20))**2)    #9Li/8He 
  #      chi2s.append((0.1/(0.1*100))**2)     #Fast neutrons 
  #      chi2s.append((0.05/(0.05*0.50))**2)  #13C(α,n)16O 
  #      chi2s.append((1/(1*0.02))**2)        #Global reactors (L > 300 km) 
  #      chi2s.append((0.16/(0.16*0.50))**2)  #Atmospheric ν’s 
  #      
  #      chi2s.append( (1/0.02)**2)  #Reactor-related correlated  correlated
  #      chi2s.append( (1/0.008)**2) #Reactor-related uncorrelated correlated
  #      chi2s.append( (1/0.30)**2)  #Spent nuclear fuel (SNF) rate  correlated
  #      chi2s.append( (1/0.30)**2)  #Non-equilibrium rate correlated
  #      chi2s.append( (1/0.06)**2)  #Matter density correlated
  #      chi2s.append( (1/0.01)**2)  #Detector related normalization correlated
  #      chi2s.append( (1/0.005)**2) #Liquid scintillator nonlinearity (LSNL)

        total_chi2 = sum(chi2s)

        if print_chi2:
            print('chi2s = {}'.format(chi2s))
            print('stat. chi2 = {}'.format(sum(chi2s[:self.spectrum_count])))
            print('syst. chi2 = {}'.format(sum(chi2s[self.spectrum_count:])))
            print('total_chi2 = {}'.format(total_chi2))

        return total_chi2
def set_legend_style(lg, **kwargs):
    text_size = kwargs.get('text_size', 28)
    lg.SetTextFont(43)
    lg.SetTextSize(text_size)
    lg.SetFillStyle(0)
    lg.SetMargin(0.4)
    lg.SetBorderSize(0)
def set_graph_style(gr, **kwargs):
    label_title_size = kwargs.get('label_title_size', 28)

    gr.SetMarkerStyle(24)
    gr.SetMarkerSize(1.2)
    gr.SetMarkerColor(kBlack)

    axiss = [gr.GetXaxis(), gr.GetYaxis()]
    if type(gr) == TGraph2D:
        axiss = [gr.GetXaxis(), gr.GetYaxis(), gr.GetZaxis()]

    for axis in axiss:
        axis.SetTitleOffset(1.2)
        axis.CenterTitle()
        axis.SetTitleFont(43)
        axis.SetLabelFont(43)
        axis.SetLabelSize(label_title_size)
        axis.SetTitleSize(label_title_size)
    # gr.GetYaxis().SetDecimals() # same label width
    # gr.GetYaxis().SetMaxDigits(3) # changes power of 10
    gr.SetLineWidth(2)
    gr.SetTitle('')

def fit_nmo_chi2_simultaneous(exposure):
 #   th12 = 33.4           # deg
 #   th13 = 8.6            # deg
 #   dm2_s = 7.4           # e-5 eV^2, small mass splitting
 #   dm2_l = 2.5           # e-3 eV^2, large mass splitting

    th12 = 33.65  #arcsin(sqrt(0.307))
    th13 = 8.491  #arcsin(sqrt(2.18*10^(-2)))
    dm2_s = 7.53
    dm2_l  = 2.546  
    dm2_l_data = 2.453 
    bin_width = 0.001

    h_smearing = None
    exposure = exposure#6 #+(1/12*(4/4))#+(1/12*(1/4))
    
    
    #tf_leakagePV1 = TFile('data/Leak_matrix_5088_5200.bin_width_{}.root'.format(bin_width))      #load the smearing matrix for reco->true
    #h_leakagePV1 = tf_leakagePV1.Get('h2_LeakMatrix') 
    #h_leakagePV1.SetDirectory(0)
    #tf_leakagePV1.Close()
    
    #tf_leakagePV2 = TFile('data/Leak_matrix_5200_5400.bin_width_{}.root'.format(bin_width))      #load the smearing matrix for reco->true
    #h_leakagePV2 = tf_leakagePV2.Get('h2_LeakMatrix') 
    #h_leakagePV2.SetDirectory(0)
    #tf_leakagePV2.Close()
    
    #tf_leakagePV3 = TFile('data/Leak_matrix_5400_5545.bin_width_{}.root'.format(bin_width))      #load the smearing matrix for reco->true
    #h_leakagePV3 = tf_leakagePV3.Get('h2_LeakMatrix') 
    #h_leakagePV3.SetDirectory(0)
    #tf_leakagePV3.Close()
    
    
    #h_leakages = [None, h_leakagePV1, h_leakagePV2, h_leakagePV3]
    h_leakages = [None, None, None, None]
######################  
##Ak FV
    #a = 2.681240 / 100
    #b = 0.7596468 / 100 
    #c = 0.9982543 / 100
    aerr = 0.01500648
    berr = 0.007532366 
    cerr = 0.03469338
    
    
##JUNO abc#
#    a= 2.72 / 100
#    b = 0.83 / 100
#    c = 1.23 / 100

##IHEP
    a = 2.61 / 100
    b = 0.82 / 100
    c = 1.23 / 100

##Eres Smear PV
    df = pd.read_csv("data/Eresparam_5088.txt", sep=",").set_index("i")
    
    dfRFV = df.loc[df.Rmin<5200]
    print("a",dfRFV.a.mean()/100, "b",dfRFV.b.mean()/100, "c",dfRFV.c.mean()/100)
    #a = 0.02637340730769231 
    #b = 0.007532724346153846 
    #c = 0.010679136269230768

    
    dfRPV1=df.loc[df.Rmin==5088]
    a_PV1 = dfRPV1.a.mean()/100
    aerr_PV1 = dfRPV1["a-err"].mean()
    b_PV1 = dfRPV1.b.mean()/100
    berr_PV1 = dfRPV1["b-err"].mean()
    c_PV1 = dfRPV1.c.mean()/100
    cerr_PV1 = dfRPV1["c-err"].mean()

    dfRPV2=df.loc[df.Rmin==5200]
    a_PV2 = dfRPV2.a.mean()/100
    aerr_PV2 = dfRPV2["a-err"].mean()
    b_PV2 = dfRPV2.b.mean()/100
    berr_PV2 = dfRPV2["b-err"].mean()
    c_PV2 = dfRPV2.c.mean()/100
    cerr_PV2 = dfRPV2["c-err"].mean()
    
    dfRPV3=df.loc[df.Rmin==5400]
    a_PV3 = dfRPV3.a.mean()/100
    aerr_PV3 = dfRPV3["a-err"].mean()
    b_PV3 = dfRPV3.b.mean()/100
    berr_PV3 = dfRPV3["b-err"].mean()
    c_PV3 = dfRPV3.c.mean()/100
    cerr_PV3 = dfRPV3["c-err"].mean()

    
    res_abcs=[[a,b,c],[a_PV1,b_PV1,c_PV1],[a_PV2,b_PV2,c_PV2],[a_PV3,b_PV3,c_PV3]]
    res_abcs_err = [[aerr,berr,cerr],[aerr_PV1,berr_PV1,cerr_PV1],[aerr_PV2,berr_PV2,cerr_PV2],[aerr_PV3,berr_PV3,cerr_PV3]]
    
    ###CV abcs##########
    #res_abcs=[[a,b,c],[a,b,c],[a,b,c],[a,b,c]]
    #res_abcs_err = [[aerr,berr,cerr],[aerr_PV1,berr_PV1,cerr_PV1],[aerr_PV2,berr_PV2,cerr_PV2],[aerr_PV3,berr_PV3,cerr_PV3]]
    ##################

    
    ### NUISANCE ###
    ################
    a_res = a
    b_res = b
    c_res = c
    
    nuisance_p = [a_res,b_res,c_res] 
    
    signal_scalings = [
    1,       #CV
    1,#      
    #    0.0220, #PV1
    1,#      
   #     0.0393, #PV2
     1#      
    #0.0285  #PV3
    ]
    
    
    geo_unc = 0
    li9_unc = 0
    alpha_unc = 0
    fast_unc = 0
    world_unc = 0
    acc_unc = 0
    
    alpha_SNF = 0
    alpha_NonEq = 0
    
    alpha_CorRea = 0
    alpha_Det = 0
    
  #  alpha_YJ1 = 0
  #  alpha_YJ2 = 0
  #  alpha_YJ3 = 0
  #  alpha_YJ4 = 0
  #  alpha_YJ5 = 0
  #  alpha_YJ6 = 0
  #  alpha_TS1 = 0
  #  alpha_TS2 = 0
  #  alpha_DYB = 0
    
    alpha_YJ = 0
    alpha_TS = 0
    alpha_DYB = 0
    alpha_HZ = 0
    
    selection_efficiency= [
    #Vcut     Ecut     deltaT   Rp-d     Muon
    #  0.822,0.822,0.822,0.822
    0.915   * 0.9983 * 0.9902 * 0.9923 * 0.916, #= 0.822  #CV 
    0.02136 * 0.9983 * 0.9902 * 0.9923 * 0.916, #= 0.009  #AV1
    0.038   * 0.9983 * 0.9902 * 0.9923 * 0.916, #= 0.016  #AV2
    0.01754 * 0.9983 * 0.9902 * 0.9923 * 0.916  #= 0.014  #AV3
    ]
    
    BG_CV = load_background(Ri=0, Rf=5000, Scale = 1, exposure=exposure)
    BG_AV1 = load_background(Ri=5088, Rf=5200, Scale = 0.0220, exposure=exposure)
    BG_AV2 = load_background(Ri=5200, Rf=5400, Scale = 0.0393, exposure=exposure)
    BG_AV3 = load_background(Ri=5400, Rf=5545, Scale = 0.0285, exposure=exposure)
    
    h_backgrounds_nom =[BG_CV,BG_AV1,BG_AV2,BG_AV3]
    

    h_FV = get_fake_data(
            th12=th12, th13=th13, dm2_s=dm2_s, dm2_l=dm2_l_data,
            a_res = a_res,
            b_res = b_res,
            c_res = c_res,
            is_normal_ordering=True,
            h_smearing=h_smearing,
            res_abc=res_abcs[0],
            h_leakage=h_leakages[0],
            bin_width=bin_width,
            selection_efficiency = selection_efficiency[0],
            exposure = exposure,
            alpha_NonEq = alpha_NonEq, 
            alpha_SNF = alpha_SNF)
    #h_PV1 = get_fake_data(
    #        th12=th12, th13=th13, dm2_s=dm2_s, dm2_l=dm2_l_data,
    #        a_res = res_abcs[1][0],
    #        b_res = res_abcs[1][1],
    #        c_res = res_abcs[1][2],
    #        is_normal_ordering=False,
    #        h_smearing=h_smearing,
    #        res_abc=res_abcs[1],
    #        h_leakage=h_leakages[1],
    #        bin_width=bin_width,
    #        signal_scaling=signal_scalings[1],
    #        selection_efficiency = selection_efficiency[1],
    #        exposure = exposure,
    #        alpha_NonEq = alpha_NonEq, 
    #        alpha_SNF = alpha_SNF)
    #h_PV2 = get_fake_data(
    #        th12=th12, th13=th13, dm2_s=dm2_s, dm2_l=dm2_l_data,
    #        a_res = res_abcs[2][0],
    #       b_res = res_abcs[2][1],
    #        c_res = res_abcs[2][2],
    #        is_normal_ordering=False,
    #        h_smearing=h_smearing,
    #        res_abc=res_abcs[2],
    #        h_leakage=h_leakages[2],
    #        bin_width=bin_width,
    #        signal_scaling=signal_scalings[2],
    #        selection_efficiency = selection_efficiency[2],
    #        exposure = exposure,
    #        alpha_NonEq = alpha_NonEq, 
    #        alpha_SNF = alpha_SNF
    #        )
    #h_PV3 = get_fake_data(
    #        th12=th12, th13=th13, dm2_s=dm2_s, dm2_l=dm2_l_data,
    #        a_res = res_abcs[3][0],
    #        b_res = res_abcs[3][1],
    #        c_res = res_abcs[3][2],
    #        is_normal_ordering=False,
    #        h_smearing=h_smearing,
    #        res_abc=res_abcs[3],
    #        h_leakage=h_leakages[3],
    #        bin_width=bin_width,
    #        signal_scaling=signal_scalings[3],
    #        selection_efficiency = selection_efficiency[3],
    #        exposure = exposure,
    #        alpha_NonEq = alpha_NonEq, 
    #        alpha_SNF = alpha_SNF)
                
    h_datas = [
        h_FV]#,
        #h_PV1,
        #h_PV2,
        #h_PV3
   #]
###Add BG to data
#    for j, h_data in enumerate(h_datas):
#        for i in range(1, h_data.GetNbinsX() + 1):
#            geo_BG = h_backgrounds_nom[j][0].GetBinContent(i)
#            li9_BG = h_backgrounds_nom[j][1].GetBinContent(i) 
#            alpha_BG = h_backgrounds_nom[j][2].GetBinContent(i)
#            fast_BG = h_backgrounds_nom[j][3].GetBinContent(i) 
#            world_BG = h_backgrounds_nom[j][4].GetBinContent(i) 
#            acc_BG = h_backgrounds_nom[j][5].GetBinContent(i)
#            h_data.SetBinContent(i, h_data.GetBinContent(i)+(geo_BG+li9_BG+alpha_BG+fast_BG+world_BG+acc_BG))
    
    cost_function = NmoCostFunctionSimultaneousFit(h_smearing, h_leakages, res_abcs, h_datas, h_backgrounds_nom, signal_scalings, res_abcs_err, selection_efficiency, exposure)
    is_normal_ordering = False
    m = Minuit(
        cost_function,
        th12=th12,
        th13=th13,
        dm2_s=dm2_s,
        dm2_l=dm2_l,
        a_res = a_res,
        b_res = b_res,
        c_res = c_res,
        a_res_AV1 = res_abcs[1][0],
        b_res_AV1 = res_abcs[1][1],
        c_res_AV1 = res_abcs[1][2],
        
        a_res_AV2 = res_abcs[2][0],
        b_res_AV2 = res_abcs[2][1],
        c_res_AV2 = res_abcs[2][2],
        
        a_res_AV3 = res_abcs[3][0],
        b_res_AV3 = res_abcs[3][1],
        c_res_AV3 = res_abcs[3][2],
        
        eff_CV = selection_efficiency[0],
        eff_AV1 = selection_efficiency[1],       
        eff_AV2 = selection_efficiency[2],
        eff_AV3 = selection_efficiency[3],
        
        geo_unc = geo_unc, 
        li9_unc = li9_unc, 
        alpha_unc = alpha_unc, 
        fast_unc = fast_unc, 
        world_unc = world_unc,
        acc_unc = acc_unc,
        
        alpha_SNF = alpha_SNF,
        alpha_NonEq = alpha_NonEq,
        
        alpha_CorRea = alpha_CorRea,
        alpha_Det = alpha_Det,
        
        alpha_YJ = alpha_YJ,
        alpha_TS = alpha_TS,
        alpha_DYB = alpha_DYB,
        alpha_HZ = alpha_HZ,
        
               
        is_normal_ordering=is_normal_ordering
    )
    m.errordef = Minuit.LEAST_SQUARES
#    m.limits['a_res'] = (a-a*0.01500648, a + a*0.01500648)
#    m.limits['b_res'] = (b-b*0.007532366, b + b*0.007532366)
#    m.limits['c_res'] = (c-c*0.03469338, c + c*0.03469338)
#  
#    m.limits['a_res_AV1'] = (res_abcs[1][0] - res_abcs[1][0]*res_abcs_err[1][0], res_abcs[1][0] + res_abcs[1][0]*res_abcs_err[1][0])
#    m.limits['b_res_AV1'] = (res_abcs[1][1] - res_abcs[1][1]*res_abcs_err[1][1], res_abcs[1][1] + res_abcs[1][1]*res_abcs_err[1][1])
#    m.limits['c_res_AV1'] = (res_abcs[1][2] - res_abcs[1][2]*res_abcs_err[1][2], res_abcs[1][2] + res_abcs[1][2]*res_abcs_err[1][2]) 
#    
#    m.limits['a_res_AV2'] = (res_abcs[2][0] - res_abcs[2][0]*res_abcs_err[2][0], res_abcs[2][0] + res_abcs[2][0]*res_abcs_err[2][0])
#    m.limits['b_res_AV2'] = (res_abcs[2][1] - res_abcs[2][1]*res_abcs_err[2][1], res_abcs[2][1] + res_abcs[2][1]*res_abcs_err[2][1])
#    m.limits['c_res_AV2'] = (res_abcs[2][2] - res_abcs[2][2]*res_abcs_err[2][2], res_abcs[2][2] + res_abcs[2][2]*res_abcs_err[2][2])
#        
#    m.limits['a_res_AV3'] = (res_abcs[3][0] - res_abcs[3][0]*res_abcs_err[3][0], res_abcs[3][0] + res_abcs[3][0]*res_abcs_err[3][0])
#    m.limits['b_res_AV3'] = (res_abcs[3][1] - res_abcs[3][1]*res_abcs_err[3][1], res_abcs[3][1] + res_abcs[3][1]*res_abcs_err[3][1])
#    m.limits['c_res_AV3'] = (res_abcs[3][2] - res_abcs[3][2]*res_abcs_err[3][2], res_abcs[3][2] + res_abcs[3][2]*res_abcs_err[3][2])
#    
#    m.limits['eff_CV'] = (selection_efficiency[0] - 0.00427, selection_efficiency[0] + 0.00427)
#    m.limits['eff_AV1'] = (selection_efficiency[1] - 0.00065, selection_efficiency[1] + 0.00065)
#    m.limits['eff_AV2'] = (selection_efficiency[2] - 0.00087, selection_efficiency[2] + 0.00087)
#    m.limits['eff_AV3'] = (selection_efficiency[3] - 0.00059, selection_efficiency[3] + 0.00059)
    
    m.fixed['is_normal_ordering'] = True
    
    m.fixed['geo_unc'] = True   #83.34402649644036
    m.fixed['li9_unc'] = True  #1276328790.0895238
    m.fixed['alpha_unc'] = True  #31046.23235568047
    m.fixed['fast_unc'] = True  #1045452721.1158472
    m.fixed['world_unc'] = True  #6489281907.633851
    m.fixed['acc_unc'] = True
    
    
    m.fixed['a_res'] = True
    m.fixed['b_res'] = True
    m.fixed['c_res'] = True
    m.fixed['a_res_AV1'] = True
    m.fixed['b_res_AV1'] = True
    m.fixed['c_res_AV1'] = True
        
    m.fixed['a_res_AV2'] = True
    m.fixed['b_res_AV2'] = True
    m.fixed['c_res_AV2'] = True
        
    m.fixed['a_res_AV3'] = True
    m.fixed['b_res_AV3'] = True
    m.fixed['c_res_AV3'] = True
        
    m.fixed['eff_CV'] = True
    m.fixed['eff_AV1'] = True       
    m.fixed['eff_AV2'] = True
    m.fixed['eff_AV3'] = True
        
    m.fixed['alpha_SNF'] = True
    m.fixed['alpha_NonEq'] = True
        
    m.fixed['alpha_CorRea'] = True
    m.fixed['alpha_Det'] = True
    
    m.fixed['alpha_YJ'] = True
    m.fixed['alpha_TS'] = True
    m.fixed['alpha_DYB'] = True
    m.fixed['alpha_HZ'] = True
    
    
#    m.limits['geo_unc'] = (-0.05,0.05)
#    m.limits['li9_unc'] =  (-0.10,0.10)
#    m.limits['alpha_unc'] = (-0.50,0.50)
#    m.limits['fast_unc'] = (-0.20,0.20)
#    m.limits['world_unc'] = (-0.05,0.05)
    
#    m.limits['geo_unc'] = (-0.0001,0.0001)
#    m.limits['li9_unc'] =  (-0.0001,0.0001)
#    m.limits['alpha_unc'] = (-0.0001,0.0001)
#    m.limits['fast_unc'] = (-0.0001,0.0001)
#    m.limits['world_unc'] = (-0.0001,0.0001)
    
    m.migrad()
    m.hesse()
    print(m.params)

    chi2 = cost_function(
        m.params['th12'].value,
        m.params['th13'].value,
        m.params['dm2_s'].value,
        m.params['dm2_l'].value,
        m.params['a_res'].value,
        m.params['b_res'].value,
        m.params['c_res'].value,
        
        m.params['a_res_AV1'].value,
        m.params['b_res_AV1'].value,
        m.params['c_res_AV1'].value,
        
        m.params['a_res_AV2'].value,
        m.params['b_res_AV2'].value,
        m.params['c_res_AV2'].value,
        
        m.params['a_res_AV3'].value,
        m.params['b_res_AV3'].value,
        m.params['c_res_AV3'].value,
        
        m.params['eff_CV'].value,
        m.params['eff_AV1'].value,
        m.params['eff_AV2'].value,
        m.params['eff_AV3'].value,    
        
        m.params['geo_unc'].value,
        m.params['li9_unc'].value,
        m.params['alpha_unc'].value,
        m.params['fast_unc'].value,
        m.params['world_unc'].value,
        m.params['acc_unc'].value,
        
        m.params['alpha_NonEq'].value,
        m.params['alpha_SNF'].value,
        
        m.params['alpha_CorRea'].value,
        m.params['alpha_Det'].value,
        
        m.params['alpha_YJ'].value,
        m.params['alpha_TS'].value,
        m.params['alpha_DYB'].value,
        m.params['alpha_HZ'].value,
        
        m.params['is_normal_ordering'].value,
        print_chi2=True
    )

    init_chi2 = cost_function(
        m.init_params['th12'].value,
        m.init_params['th13'].value,
        m.init_params['dm2_s'].value,
        m.init_params['dm2_l'].value,
        m.init_params['a_res'].value,
        m.init_params['b_res'].value,
        m.init_params['c_res'].value,
        
        m.init_params['a_res_AV1'].value,
        m.init_params['b_res_AV1'].value,
        m.init_params['c_res_AV1'].value,
        
        m.init_params['a_res_AV2'].value,
        m.init_params['b_res_AV2'].value,
        m.init_params['c_res_AV2'].value,
        
        m.init_params['a_res_AV3'].value,
        m.init_params['b_res_AV3'].value,
        m.init_params['c_res_AV3'].value,
        
        m.init_params['eff_CV'].value,
        m.init_params['eff_AV1'].value,
        m.init_params['eff_AV2'].value,
        m.init_params['eff_AV3'].value,  
        
        m.init_params['geo_unc'].value,
        m.init_params['li9_unc'].value,
        m.init_params['alpha_unc'].value,
        m.init_params['fast_unc'].value,
        m.init_params['world_unc'].value,
        m.init_params['acc_unc'].value,
        
        m.init_params['alpha_NonEq'].value,
        m.init_params['alpha_SNF'].value,
        
        m.init_params['alpha_CorRea'].value,
        m.init_params['alpha_Det'].value,
        
        m.init_params['alpha_YJ'].value,
        m.init_params['alpha_TS'].value,
        m.init_params['alpha_DYB'].value,
        m.init_params['alpha_HZ'].value,
        
        m.init_params['is_normal_ordering'].value,
        print_chi2=False
    )

    print('chi2 = {}'.format(chi2))
    print('init_chi2 = {}'.format(init_chi2))

##########################################################################
#    h_FVinit = get_fake_data(
#            th12=m.init_params['th12'].value,
#            th13=m.init_params['th13'].value,
#            dm2_s=m.init_params['dm2_s'].value,
#            dm2_l=m.init_params['dm2_l'].value,
#            a_res=m.init_params['a_res'].value,
#            b_res=m.init_params['b_res'].value,
#            c_res=m.init_params['c_res'].value,
#            is_normal_ordering=m.init_params['is_normal_ordering'].value,
#            h_smearing=h_smearing,
#            res_abc=res_abcs[0],
#            h_leakage=h_leakages[0],
#            bin_width=bin_width,
#            h_background=scale_background(BG=h_backgrounds_nom[0],geo_unc = m.init_params['geo_unc'].value, li9_unc = m.init_params['li9_unc'].value, alpha_unc = m.init_params['alpha_unc'].value,fast_unc = m.init_params['fast_unc'].value, world_unc = m.init_params['world_unc'].value),
#            signal_scaling=signal_scalings[0],
#            selection_efficiency = m.init_params['eff_CV'].value,
#            exposure = exposure,
#            alpha_NonEq = m.init_params['alpha_NonEq'].value, 
#            alpha_SNF = m.init_params['alpha_SNF'].value
#            )
#    
#    h_PV1init = get_fake_data(
#            th12=m.init_params['th12'].value,
#            th13=m.init_params['th13'].value,
#            dm2_s=m.init_params['dm2_s'].value,
#            dm2_l=m.init_params['dm2_l'].value,
#        
#            a_res = m.init_params['a_res_AV1'].value,
#            b_res = m.init_params['b_res_AV1'].value,
#            c_res = m.init_params['c_res_AV1'].value,
#
#            is_normal_ordering=m.init_params['is_normal_ordering'].value,
#            h_smearing=h_smearing,
#            res_abc=res_abcs[1],
#            h_leakage=h_leakages[1],
#            bin_width=bin_width,
#            h_background=scale_background(BG=h_backgrounds_nom[1],geo_unc = m.init_params['geo_unc'].value,li9_unc = m.init_params['li9_unc'].value,alpha_unc = m.init_params['alpha_unc'].value,fast_unc = m.init_params['fast_unc'].value,world_unc = m.init_params['world_unc'].value),
#            signal_scaling=signal_scalings[1],
#            selection_efficiency = m.init_params['eff_AV1'].value,
#            exposure = exposure,
#            alpha_NonEq = m.init_params['alpha_NonEq'].value, 
#            alpha_SNF = m.init_params['alpha_SNF'].value)
#    
#    h_PV2init = get_fake_data(
#            th12=m.init_params['th12'].value,
#            th13=m.init_params['th13'].value,
#            dm2_s=m.init_params['dm2_s'].value,
#            dm2_l=m.init_params['dm2_l'].value,
#            a_res = m.init_params['a_res_AV2'].value,
#            b_res = m.init_params['b_res_AV2'].value,
#            c_res = m.init_params['c_res_AV2'].value,
#            is_normal_ordering=m.init_params['is_normal_ordering'].value,
#            h_smearing=h_smearing,
#            res_abc=res_abcs[2],
#            h_leakage=h_leakages[2],
#            bin_width=bin_width,
#            h_background=scale_background(BG=h_backgrounds_nom[2],geo_unc = m.init_params['geo_unc'].value,li9_unc = m.init_params['li9_unc'].value,alpha_unc = m.init_params['alpha_unc'].value,fast_unc = m.init_params['fast_unc'].value,world_unc = m.init_params['world_unc'].value),  
#            signal_scaling=signal_scalings[2],
#            selection_efficiency = m.init_params['eff_AV2'].value,
#            exposure = exposure,
#            alpha_NonEq = m.init_params['alpha_NonEq'].value, 
#            alpha_SNF = m.init_params['alpha_SNF'].value)
#    
#    h_PV3init = get_fake_data(
#            th12=m.init_params['th12'].value,
#            th13=m.init_params['th13'].value,
#            dm2_s=m.init_params['dm2_s'].value,
#            dm2_l=m.init_params['dm2_l'].value,
#            a_res = m.init_params['a_res_AV3'].value,
#            b_res = m.init_params['b_res_AV3'].value,
#            c_res = m.init_params['c_res_AV3'].value,
#            is_normal_ordering=m.init_params['is_normal_ordering'].value,
#            h_smearing=h_smearing,
#            res_abc=res_abcs[3],
#            h_leakage=h_leakages[3],
#            bin_width=bin_width,
#            h_background=scale_background(BG=h_backgrounds_nom[3],geo_unc = m.init_params['geo_unc'].value,li9_unc = m.init_params['li9_unc'].value,alpha_unc = m.init_params['alpha_unc'].value,fast_unc = m.init_params['fast_unc'].value,world_unc = m.init_params['world_unc'].value),
#            signal_scaling=signal_scalings[3],
#            selection_efficiency = m.init_params['eff_AV3'].value,
#            exposure = exposure,
#            alpha_NonEq = m.init_params['alpha_NonEq'].value, 
#            alpha_SNF = m.init_params['alpha_SNF'].value)
#
#    
#    h_inits = [h_FVinit, h_PV1, h_PV2, h_PV3]
#    
#    h_FVbest = get_fake_data(
#            th12=m.params['th12'].value,
#            th13=m.params['th13'].value,
#            dm2_s=m.params['dm2_s'].value,
#            dm2_l=m.params['dm2_l'].value,
#            a_res=m.params['a_res'].value,
#            b_res=m.params['b_res'].value,
#            c_res=m.params['c_res'].value,
#            is_normal_ordering=m.params['is_normal_ordering'].value,
#            h_smearing=h_smearing,
#            h_leakage=h_leakages[0],
#            res_abc=res_abcs[0],
#            bin_width=bin_width,
#            h_background=scale_background(BG=h_backgrounds_nom[0],geo_unc = m.params['geo_unc'].value,li9_unc = m.params['li9_unc'].value, alpha_unc = m.params['alpha_unc'].value,fast_unc = m.params['fast_unc'].value,world_unc = m.params['world_unc'].value),
#            signal_scaling=signal_scalings[0],
#            selection_efficiency = m.params['eff_CV'].value,
#            exposure = exposure,
#            alpha_NonEq = m.params['alpha_NonEq'].value, 
#            alpha_SNF = m.params['alpha_SNF'].value
#        )
#    
#    h_PV1best = get_fake_data(
#            th12=m.params['th12'].value,
#            th13=m.params['th13'].value,
#            dm2_s=m.params['dm2_s'].value,
#            dm2_l=m.params['dm2_l'].value,
#            a_res = m.params['a_res_AV1'].value,
#            b_res = m.params['b_res_AV1'].value,
#            c_res = m.params['c_res_AV1'].value,
#            is_normal_ordering=m.params['is_normal_ordering'].value,
#            h_smearing=h_smearing,
#            h_leakage=h_leakages[1],
#            res_abc=res_abcs[1],
#            bin_width=bin_width,
#            h_background=scale_background(BG=h_backgrounds_nom[1],geo_unc = m.params['geo_unc'].value,li9_unc = m.params['li9_unc'].value,alpha_unc = m.params['alpha_unc'].value,fast_unc = m.params['fast_unc'].value,world_unc = m.params['world_unc'].value),
#            signal_scaling=signal_scalings[1],
#            selection_efficiency = m.params['eff_AV1'].value,
#            exposure = exposure,
#            alpha_NonEq = m.params['alpha_NonEq'].value, 
#            alpha_SNF = m.params['alpha_SNF'].value
#        )
#    h_PV2best = get_fake_data(
#            th12=m.params['th12'].value,
#            th13=m.params['th13'].value,
#            dm2_s=m.params['dm2_s'].value,
#            dm2_l=m.params['dm2_l'].value,
#            a_res = m.params['a_res_AV2'].value,
#            b_res = m.params['b_res_AV2'].value,
#            c_res = m.params['c_res_AV2'].value,
#            is_normal_ordering=m.params['is_normal_ordering'].value,
#            h_smearing=h_smearing,
#            h_leakage=h_leakages[2],
#            res_abc=res_abcs[2],
#            bin_width=bin_width,
#            h_background=scale_background(BG=h_backgrounds_nom[2],geo_unc = m.params['geo_unc'].value,li9_unc = m.params['li9_unc'].value,alpha_unc = m.params['alpha_unc'].value,fast_unc = m.params['fast_unc'].value,world_unc = m.params['world_unc'].value),
#            signal_scaling=signal_scalings[2],
#            selection_efficiency = m.params['eff_AV2'].value,
#            exposure = exposure,
#            alpha_NonEq = m.params['alpha_NonEq'].value, 
#            alpha_SNF = m.params['alpha_SNF'].value
#        )
#    h_PV3best = get_fake_data(
#            th12=m.params['th12'].value,
#            th13=m.params['th13'].value,
#            dm2_s=m.params['dm2_s'].value,
#            dm2_l=m.params['dm2_l'].value,
#            a_res = m.params['a_res_AV3'].value,
#            b_res = m.params['b_res_AV3'].value,
#            c_res = m.params['c_res_AV3'].value,
#            is_normal_ordering=m.params['is_normal_ordering'].value,
#            h_smearing=h_smearing,
#            h_leakage=h_leakages[3],
#            res_abc=res_abcs[3],
#            bin_width=bin_width,
#            h_background=scale_background(BG=h_backgrounds_nom[3],geo_unc = m.params['geo_unc'].value,li9_unc = m.params['li9_unc'].value,alpha_unc = m.params['alpha_unc'].value,fast_unc = m.params['fast_unc'].value,world_unc = m.params['world_unc'].value),
#            signal_scaling=signal_scalings[3],
#            selection_efficiency = m.params['eff_AV3'].value,
#            exposure = exposure,
#            alpha_NonEq = m.params['alpha_NonEq'].value, 
#            alpha_SNF = m.params['alpha_SNF'].value
#        )
#
#    h_bests = [h_FVbest, h_PV1best, h_PV2best, h_PV3best]
#    c1 = TCanvas('c1', 'c1', 4000, 1000)
#
#    gPad.SetLeftMargin(0.15)
#    gPad.SetBottomMargin(0.15)
#    c1.Divide(4,1,0,0)
#
#    directions = [r'CV (R^{3}< 5088 m^{3})', 
#                  'AV1 (5088<R^{3}<5200 m^{3})',
#                 'AV2 (5200<R^{3}<5400 m^{3})',
#                 'AV3 (5400<R^{3}<5545 m^{3})']
#
#    maxes = [620,400,800,800]
#    gr_datas = []
#    for i in range(len(signal_scalings)):
#        c1.cd(i + 1)
#        gPad.SetLeftMargin(0.2)
#        gPad.SetBottomMargin(0.2)
#
#        h_data = h_datas[i]
#        h_background = h_backgrounds[i]
#        h_init = h_inits[i]
#        h_best = h_bests[i]
#        gr_datas.append(get_graph_from_hist(h_data))
#        gr_data = gr_datas[i]
#
#        lg1 = TLegend(0.48, 0.54, 0.8, 0.85)
#        set_legend_style(lg1)
#
#        set_graph_style(gr_data)
#        gr_data.SetMarkerStyle(20)
#        gr_data.SetMarkerSize(0.2)
#        gr_data.SetLineStyle(1)
#        gr_data.GetXaxis().SetRangeUser(0.8, 9)
#        gr_data.GetYaxis().SetRangeUser(0, maxes[i])
#
#        gr_data.GetXaxis().SetRangeUser(3, 9)
#        gr_data.GetXaxis().SetTitle('E_{reco.} (MeV)')
#        gr_data.GetYaxis().SetTitle('Event Count / {:.0f} keV'.format(bin_width * 1000))
#        gr_data.SetLineColor(kGray + 1)
#        gr_data.Draw('APE')
#        lg1.AddEntry(gr_data, 'NO Asmiov data', 'lpe')
#
#        set_h1_style(h_background)
#        h_background.SetStats(0)
#        h_background.SetLineColor(kGreen + 2)
#        h_background.Draw('sames, hist')
#        lg1.AddEntry(h_background, 'background', 'l')
#
#        set_h1_style(h_init)
#        h_init.SetStats(0)
#        h_init.SetLineColor(kBlue)
#        h_init.Draw('sames, hist')
#        lg1.AddEntry(h_init, 'inital parameters', 'l')
#
#        set_h1_style(h_best)
#        h_best.SetStats(0)
#        h_best.SetLineColor(kRed)
#        h_best.Draw('sames, hist')
#        lg1.AddEntry(h_best, 'IO best fit', 'l')
#
#        tex = TLatex()
#        tex.SetNDC()
#        tex.SetTextFont(43)
#        tex.SetTextSize(28)
#        tex.SetTextColor(kRed)
#        tex.SetTextAngle(0)
#        tex.SetTextAlign(22)
#        tex.DrawLatex(0.8, 0.95, '#chi^{{2}} = {:.1f}'.format(chi2))
#        tex.DrawLatex(0.8, 0.86, directions[i])
#        c1.Update()
#
#        lg1.Draw()
#
#    c1.Update()
#    c1.SaveAs('{}/fit_nmo_chi2_simultaneous_R3Chunks_ares.pdf'.format(FIGURE_DIR))
#    #input('Press any key to continue.')
########################################################################################
    return chi2
fit_nmo_chi2_simultaneous(exposure=6)
