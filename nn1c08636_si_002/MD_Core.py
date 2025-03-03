import pandas as pd
import numpy as np
import math
from scipy import optimize
from scipy import signal
import copy

class Mult_Diff():
       
#______________________________________________________________________________

    def __init__(self,filename):
        '''
        Write function comment here
        '''
        
        self.filename = filename                                                # Takes the data filename 
        self.data = pd.read_csv(self.filename,sep=';')                          # Reads q and I from the data file
        
        self.q = self.data.iloc[:,0].values                                     # Extracts self.qmeasured from self.data
        self.I = self.data.iloc[:,1].values/max(self.data.iloc[:,1].values)     # Extracts self.Imeasured from self.data and normalizes to 1
        
        
#______________________________________________________________________________


    def correct(self,q_range,I_range):    
        '''
        Applies intensity corrections to the experimental pattern, to turn the measured intensity into the square modulus of 
        the superlattice scattering factor FF*(q)
        '''
        
        thetam = math.radians(self.mirror_angle)
        A = np.cos(2*thetam)**2
        two_theta = 2*np.arcsin((self.wavelength * q_range)/(4 *math.pi))
        self.tt = two_theta

        # LPA correction factor as expressed in the Fullerton Paper
        LP_factor = (1-np.exp(-2*self.ABS/np.sin(two_theta/2)))*(1+np.cos(two_theta)**2*np.cos(2*thetam)**2)/np.sin(two_theta)
        self.lp=LP_factor
        
        # Applies the LPA correction and normalizes the intensities to 1
        I_corr = I_range / LP_factor
        I_corr = I_corr/max(I_corr)
            
        self.I_corr = I_corr
        
        return I_corr
    
    
#______________________________________________________________________________
    
    
    def atomic_scattering(self,elements):  
        '''
        Computes the atomic scattering (or form) factors over the measured q-range starting from tabulated constants found 
        at the link https://bit.ly/3m23uBU.
        New elements must be defined here first, by adding the following line:
        ion_const['Element Symbol'] = [constant 1, constant 2, constant 3, constant 4, constant 5, constant 6, constant 7, constant 8, constant 9]
        '''

        # Database of Ionic/Atomic form factors
        ion_const = pd.DataFrame()      
        ion_const['Pb'] = [31.0617, 0.6902, 13.0637, 2.3576, 18.442, 8.618, 5.9696, 47.2579, 13.4118]
        ion_const['Pb2+'] = [21.7886, 1.3366, 19.5682, 0.488383, 19.1406, 6.7727, 7.01107, 23.8132, 12.4734]
        ion_const['Cs+'] = [20.3524, 3.552, 19.1278, 0.3086, 10.2821, 23.7128, 0.9615, 59.4565, 3.2791]
        
        ion_const['C'] = [2.31, 20.8439, 1.02, 10.2075, 1.5886, 0.5687, 0.865, 51.6512, 0.2156]
        ion_const['N']  = [12.2126, 0.0057, 3.1322, 9.8933, 2.0125, 28.9975, 1.1663, 0.5826, -11.529]
        
        ion_const['S']  = [6.9053, 1.4679, 5.2034, 22.2151, 1.4379, 0.2536, 1.5863, 56.172, 0.8669]
        ion_const['Se']  = [17.0006, 2.4098, 5.8196, 0.2726, 3.9731, 15.2372, 4.3543, 43.8163, 2.8409]
        ion_const['O'] = [3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 0.867, 32.9089, 0.2508]
        
        ion_const['Cd2+']  = [19.1514, 0.597922, 17.2535, 6.80639, 4.47128, 20.2521, 0, 0, 5.11937]
        ion_const['F-'] = [3.6322, 5.27756, 3.51057, 14.7353, 1.26064, 0.442258, 0.940706, 47.3437, 0.653396]
        ion_const['Cl-']  = [11.4604, 0.0104, 7.1964, 1.1662, 6.2556, 18.5194, 1.6455, 47.7784, -9.5574]
        ion_const['Br-'] = [17.1718, 2.2059, 6.3338, 19.3345, 5.5754, 0.2871, 3.7272, 58.1535, 3.1776]
        ion_const['I-']  = [20.2332, 4.3579, 18.997, 0.3815, 7.8069, 29.5259, 2.8868, 84.9304, 4.0714]
        
        ion_const['Ca2+'] = [15.6348, -0.0074, 7.9518, 0.6089, 8.4372, 10.3116, 0.8537, 25.9905, -14.875]
        ion_const['Al3+'] = [4.17448, 1.93816, 3.3876, 4.14553, 1.20296, 0.228753, 0.528137, 8.28524, 0.706786]
        ion_const['Zn2+'] = [11.9719, 2.9946, 7.3862, 0.2031, 6.4668, 7.0826, 1.394, 18.0995, 0.7807]
        ion_const['Mn2+'] = [10.8061, 5.2796, 7.362, 0.3435, 3.5268, 14.343, 0.2184, 41.3235, 1.0874]
        
        ff = pd.DataFrame()        
        ff['q'] = self.q
        
        # For each element, computes the atomic scattering factors over the data q_range
        for elem in elements:
            ff[elem] = ion_const[elem][8] + ion_const[elem][0]*np.exp(-ion_const[elem][1]*(ff['q']/(4*np.pi))**2)\
                                          + ion_const[elem][2]*np.exp(-ion_const[elem][3]*(ff['q']/(4*np.pi))**2)\
                                          + ion_const[elem][4]*np.exp(-ion_const[elem][5]*(ff['q']/(4*np.pi))**2)\
                                          + ion_const[elem][6]*np.exp(-ion_const[elem][7]*(ff['q']/(4*np.pi))**2)  
        self.ff = ff
        
        return ff

    
#______________________________________________________________________________ 
    
    
    
    def crystal_generator(self,structure_input):
        '''
        Builds the atomistic model of the crystal, by taking into account chemical nature, position and partial occupancy 
        of all the atoms in the crystal. 
        '''
        
        structure = pd.DataFrame() 

        # Converts the input in a Pandas Data Frame
        for plane in range (0,len(structure_input)):
            structure[plane] =  structure_input[plane]

        self.structure = structure
        
        return structure        
        
#______________________________________________________________________________ 
    
    
    
    def FF_simulator(self,d):  
        '''
        Computes the nanocrystal form factor based on the current structural model and the input d (reference length) value. 
        '''
        # Calls the function that computes the atomic form factors for the required elements
        ff = self.atomic_scattering(self.elements)
        
        structure = self.structure
        
        # Computes the nanocrystal form factor for a given thickness
        F = pd.DataFrame()               
        F['q'] = self.q
        F['F'] = 0
        for col in structure:     
            for row in range (1,len(structure[col])):
                if isinstance(structure[col][row],int) or isinstance(structure[col][row],float):
                    pass
                else:
                    if structure[col][row] != " ":
                        F['F'] = F['F'] + structure[col][row+1]*ff[structure[col][row]]*np.exp(-1j*F['q']*d*structure[col][0])  
        
        # Computes and plots FF*
        F['F*F'] = np.real(F['F']*np.conjugate(F['F']))
        self.F = F
    
    
#______________________________________________________________________________ 
    
    
    def simulate_low(self, x0):  
        '''
        Core function of the pattern simulation. Takes as input the superlattice structural parameters and computes a simulated pattern. 
        This function is used both in the simulation and in the fitting routines of the code. 
        The suffix _low indicates that only the general parameters d, L, sigmaL, qzero and HWHM are considered at this stage. 
        '''
        
        d, L ,sigmaL, q_zero, HWHM = x0
        C_dens = 0
        C = self.C
        
        # Applies q_zero correction
        q = self.q_range
        q = q + q_zero
        
        # Loads atomic form factors and initial crystal structure
        ff = self.ff_range 
        structure_input = self.structure_input
        
        # Computes the structure for this specific calculation run
        structure = self.crystal_generator(structure_input)
        
        # Simulates the diffracted intensities and prepares the Form Factor calculation
        Icalc = np.zeros(len(q))
        F = pd.DataFrame()               
        F['q'] = q
        F['F_cry'] = 0
        F['F_org'] = 0
        
        # Computes the Nanocrystal and the Organics Form Factor
        for col in structure:     
            for row in range (1,len(structure[col])): # 1 credo sia per non prendere la colonna coordinate atomiche
                if isinstance(structure[col][row],int) or isinstance(structure[col][row],float):
                    pass
                else:
                    if structure[col][row] != " ":
                        F['F_cry'] = F['F_cry'] + structure[col][row+1]*ff[structure[col][row]]*np.exp(-1j*F['q']*d*structure[col][0])

        F['F_org']= C_dens*ff['C']/(1j * F['q']) * (np.exp(1j*F['q']*L)-1)
        
        # Computes the thickness of the inorganic part, layer to layer
        max_z = self.structure.shape[1] -1
        t_A = abs((self.structure[max_z][0] - self.structure[0][0])*d)
        
        # Computes several terms related to the Form Factors
        F_A = F['F_cry']
        FconjF_A = F_A*np.conj(F_A)
        PHI_A = np.exp(1j*t_A*q)*np.conj(F_A)
        T_A = np.exp(1j*t_A*q)  
       
        F_B = F['F_org']
        FconjF_B = F_B*np.conj(F_B)
        PHI_B = np.exp(1j*L*q)*np.conj(F_B)
        T_B = np.exp(1j*L*q)
            
        # Computes the superlattice diffracted intensity according to Fullerton et al. 
        XI = - q**2*sigmaL**2/2
        IA = C * (FconjF_A + 2*np.real(np.exp(XI)*PHI_A*F_B) + FconjF_B)
        IB = np.exp(-XI)*PHI_B*F_A/(T_A * T_B) + PHI_A*F_A/T_A + PHI_B*F_B/T_B + np.exp(XI)*PHI_A*F_B
        IC = ((C - (C+1)*np.exp(2*XI)*T_A*T_B + (np.exp(2*XI)*T_A*T_B)**(C+1)) / (1 - np.exp(2*XI)*T_A*T_B)**2 - C)
        Ij = IA + 2*np.real (IB* IC)
        Icalc = np.real (Ij)
        
        # Convolutes the calculated pattern with the instrumental response
        Iconv = np.zeros(len(q))
        
        for i in range(len(q)):
            # Fixed Width Gaussian
            Iconv = Iconv + Icalc[i]*1/(HWHM*(math.sqrt(2*math.pi)))*np.exp(-(q[i]-q)**2/(2*(HWHM)**2))
            
        # Normalizes the computed pattern
        self.I_sim = Iconv/max(Iconv)
        
        return self.I_sim
    

    
#______________________________________________________________________________ 


    
    def fit_low(self, x0 = None,bounds = None,fit_derivs = False):
        '''
        Nonlinear Least Square fitting routine.         
        The suffix _low indicates that only the general parameters d, L, sigmaL, qzero and HWHM are considered at this stage. 
        '''

        res = optimize.least_squares(self.Residuals_low,x0=x0,bounds = bounds,ftol = 10**(-12), gtol = None,xtol=None)
        self.x0 = res.x
        self.jac = res.jac
        
        return res
    
  

 #______________________________________________________________________________   
   
    

    def Residuals_low(self,x0):
        '''
        Calculates the residuals by comparing the outcome of the non-linear least squares fitting and the experimental pattern.        
        The suffix _low indicates that only the general parameters d, L, sigmaL, qzero and HWHM are considered at this stage. 
        '''
        
        res = np.array([])
        fit_derivs = True

        if fit_derivs:
            I_sim = self.simulate_low(x0)
            I_corr = self.I_corr
            self.I_corr = I_corr/max(I_corr)
            
            d_sim = signal.savgol_filter(I_sim,15,3,deriv=1)
            d_sim = d_sim/max(d_sim)
            d_int = signal.savgol_filter(self.I_corr,15,3,deriv=1)
            d_int = d_int/max(d_int)
            res = np.hstack((res,np.array(self.I_corr-self.I_sim + d_int-d_sim)))
            
            return res
        
        else:
            self.I_sim = self.simulate_low(x0)
            I_corr = self.I_corr
            self.I_corr = I_corr/max(I_corr)
            res = self.I_corr-self.I_sim
            
        return res

    
    
#______________________________________________________________________________ 


    
    def fit_high(self, x0 = None,bounds = None,fit_derivs = False):
        '''
        Nonlinear Least Square fitting routine.         
        The suffix _high indicates that all the fittable parameters are considered at this stage. 
        '''

        res = optimize.least_squares(self.Residuals_high,x0=x0,bounds = bounds,ftol = 10**(-12), gtol = None,xtol=None)
        self.x0 = res.x
        self.jac = res.jac
        
        return res
    
  

 #______________________________________________________________________________   
   
    

    def Residuals_high(self,x0):
        '''
        Calculates the residuals by comparing the outcome of the non-linear least squares fitting and the experimental pattern.        
        The suffix _high indicates that all the fittable parameters are considered at this stage. 
        '''

        res = np.array([])
        fit_derivs = True

        if fit_derivs:
            I_sim = self.simulate_high(x0)
            I_corr = self.I_corr
            self.I_corr = I_corr/max(I_corr)
            
            d_sim = signal.savgol_filter(I_sim,15,3,deriv=1)
            d_sim = d_sim/max(d_sim)
            d_int = signal.savgol_filter(self.I_corr,15,3,deriv=1)
            d_int = d_int/max(d_int)
            res = np.hstack((res,np.array(self.I_corr-self.I_sim + d_int-d_sim)))
            
            return res
        
        else:
            self.I_sim = self.simulate_high(x0)
            I_corr = self.I_corr
            self.I_corr = I_corr/max(I_corr)
            res = self.I_corr-self.I_sim
            
        return res
    
    
#______________________________________________________________________________ 


    
    def simulate_high(self, x0):  
        '''
        Core function of the pattern simulation. Takes as input the superlattice structural parameters and computes a simulated pattern. 
        This function is used both in the simulation and in the fitting routines of the code. 
        The suffix _high indicates that all the fittable parameters are considered at this stage. 
        '''
        
        if self.add_var == 0:    
            d, L ,sigmaL, q_zero, HWHM, C_dens = x0
            v1 = self.starting_guess[6]
            v2 = self.starting_guess[7]
            v3 = self.starting_guess[8]
            v4 = self.starting_guess[9]
        elif self.add_var == 1:    
            d, L ,sigmaL, q_zero, HWHM, C_dens,v1= x0
            v2 = self.starting_guess[7]
            v3 = self.starting_guess[8]
            v4 = self.starting_guess[9]
        elif self.add_var == 2:    
            d, L ,sigmaL, q_zero, HWHM, C_dens,v1,v2= x0
            v3 = self.starting_guess[8]
            v4 = self.starting_guess[9]
        elif self.add_var == 3:    
            d, L ,sigmaL, q_zero, HWHM, C_dens,v1,v2,v3= x0
            v4 = self.starting_guess[9]
        else:
            d, L ,sigmaL, q_zero, HWHM, C_dens,v1,v2,v3,v4 = x0
            
        C = self.C
        
        # Applies q_zero correction
        q = self.q_range
        q = q + q_zero
        
        # Loads the atomic form factors over the q-range required for the simulation
        ff = self.ff_range 

        # Loads the structural information
        structure_input = self.structure_input
        
        # Sends the structure to the relations customizer
        structure_input = self.relations(structure_input,v1,v2,v3,v4)
        
        # Compute the structure for this specific calculation run
        structure = self.crystal_generator(structure_input)
        
        # Simulates the diffracted intensities and prepares the Form Factor calculation
        Icalc = np.zeros(len(q))
        F = pd.DataFrame()               
        F['q'] = q
        F['F_cry'] = 0
        F['F_org'] = 0
        
        # Computes the Nanocrystal and the Organics Form Factor
        for col in structure:     
            for row in range (1,len(structure[col])): # 1 credo sia per non prendere la colonna coordinate atomiche
                if isinstance(structure[col][row],int) or isinstance(structure[col][row],float):
                    pass
                else:
                    if structure[col][row] != " ":
                        F['F_cry'] = F['F_cry'] + structure[col][row+1]*ff[structure[col][row]]*np.exp(-1j*F['q']*d*structure[col][0])

        F['F_org']= C_dens*ff['C']/(1j * F['q']) * (np.exp(1j*F['q']*L)-1)
        
        # Computes the thickness of the inorganic part, layer to layer
        max_z = self.structure.shape[1] -1
        t_A = abs((self.structure[max_z][0] - self.structure[0][0])*d)
        
        # Computes several terms related to the Form Factors
        F_A = F['F_cry']
        FconjF_A = F_A*np.conj(F_A)
        PHI_A = np.exp(1j*t_A*q)*np.conj(F_A)
        T_A = np.exp(1j*t_A*q)  
       
        F_B = F['F_org']
        FconjF_B = F_B*np.conj(F_B)
        PHI_B = np.exp(1j*L*q)*np.conj(F_B)
        T_B = np.exp(1j*L*q)
            
        # Computes the superlattice diffracted intensity according to Fullerton et al. 
        XI = - q**2*sigmaL**2/2
        IA = C * (FconjF_A + 2*np.real(np.exp(XI)*PHI_A*F_B) + FconjF_B)
        IB = np.exp(-XI)*PHI_B*F_A/(T_A * T_B) + PHI_A*F_A/T_A + PHI_B*F_B/T_B + np.exp(XI)*PHI_A*F_B
        IC = ((C - (C+1)*np.exp(2*XI)*T_A*T_B + (np.exp(2*XI)*T_A*T_B)**(C+1)) / (1 - np.exp(2*XI)*T_A*T_B)**2 - C)
        Ij = IA + 2*np.real (IB* IC)
        Icalc = np.real (Ij)
        
        # Convolutes the calculated pattern with the instrumental response
        Iconv = np.zeros(len(q))
           
        for i in range(len(q)):
            # Fixed Width Gaussian
            Iconv = Iconv + Icalc[i]*1/(HWHM*(math.sqrt(2*math.pi)))*np.exp(-(q[i]-q)**2/(2*(HWHM)**2))
            
        # Normalizes the computed pattern
        self.I_sim = Iconv/max(Iconv)
        
        return self.I_sim
    
    
#______________________________________________________________________________



    def fit_bootstrap(self,x0 = None,bounds = None, n_random = 100,assumed_error=0.03):
        '''
        Bootstrap routine, uses the bootstrap statistical analysis to calculate the mean value of each fittable parameter and estimate its standad deviation.  
        '''
        
        self.I_backup = copy.deepcopy(self.I_corr)
        
        results = []
        i = 0
        n_failed = 0

        while i < n_random:
                
            self.I_corr = np.random.normal(self.I_corr,scale = assumed_error,size = len(self.I_corr) )

            try:
                res = optimize.least_squares(self.Residuals_high,x0=x0,bounds = bounds,ftol = 10**(-12), gtol = None,xtol=None)
                    
            except np.linalg.LinAlgError:
                n_failed +=1
                continue
                    
            results.append([])
            self.x0 = res.x
            results[i].append(list(res.x) + list(res.fun))

            if n_failed > 100:
                print('This fitting procedure has failed too often. Try using a more reasonable starting guess, or a more reasonable error approximation')
                break

            i+=1
            print(i, end = "\r")
            self.I_corr = copy.deepcopy(self.I_backup)

        return results
        

#______________________________________________________________________________        
        
        
# NOTE:
# https://nedcharles.com/regression/Nonlinear_Least_Squares_Regression_For_Python.html
# https://stackoverflow.com/questions/7126190/standard-error-in-non-linear-regression
# https://stackoverflow.com/questions/42388139/how-to-compute-standard-deviation-errors-with-scipy-optimize-least-squares

