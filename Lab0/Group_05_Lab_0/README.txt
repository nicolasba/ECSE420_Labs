Steps to run the project using Visual Studio:

- After opening the solution: we have two .cu files that contain a main() function and are included in the project (rectify.cu and pool.cu).
In order to run rectify.cu, exclude pool.cu from the project by right-clicking on the file in the Solution explorer tab, then "Exclude from
project". In order to run pool.cu, exclude rectify.cu. If files do not appear, click on "Show all files".

- In order to pass arguments, right click on "ECSE420_Lab0" in the Solution explorer tab. Then Properties->Configuration Properties
->Debugging. Type the arguments in the "Command Arguments" line.

Alternatively each file could be compiled separately using the nvcc compiler.