This python code is for running boson coherent states field theory. 


To run the code, simply run the "main.py" script as follows in the terminal:

$ python3 main.py 
or
$ ipython3 main.py 


Please edit the main.py script directly to change the inputs. Main.py is thoroughly commented with descriptions of each input and expected output behavior. The code will give messages at the iofrequency, stating that X many steps have been completed. At the end of the simulation or after a divergence occurs, the program will plot the observables (N, the particle number) at various CL times, showing the sampling procedure. For the plotting to work, please see the import statements at the top of the CL_Driver.py class. One of the two following options are provided for making the plots show up correctly: 
 
//matplotlib.rcParams['text.usetex'] = True ## I use this on my mac osx 
//matplotlib.use('TkAgg') # use this on linux 

Notes: 
The boolean "_isShifting" in main.py will enable us to shift the linear coefficients A(n) --> A(n) + B(n), where we choose B(n) to improve stability. The choice of B(n) can be found in Bosefluid_Model.py in the "fill_forces()" function. The nonlinear force is filled in that function, while the linear coefficient A_n is filled elsewhere in fill_lincoefs() in the Bosefluid_Model class.

Increasing the repulsion strength g > 1 will often result in divergences, where the code stops because the field values are blowing up to \pm infinity. 




The code is structured as follows:

The main.py script will create the model object "Bosefluid_Model" and then create the CL Driver object. The model's forces are detailed in the Bosefluid_Model class, while the timestepping details are in Timesteppers.py and are called by the CL_Driver class. The CL_Driver class is responsible for running the main CL loop. 

The Operator class provides an object that contains the observables of interest. For this simple code, we only consider the particle number observable, since we can fully test the code with just that operator. 

 
