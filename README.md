# fermi_level
project for detection of fermi level using arpes data through machine learning.

For a high level understanding of how the programme works, refer to technical details.docx

To run inference through the programme,

run
python inference.py --filename <path to file> --energy_axis <energy_axis> --momentum_axis <momentum_axis>
Here energy axis and momentum_axis denote the axis in which these data are stored(x, y or z). An example of doing this is shown in quicktesting.png. 
Note that if you do not give energy_axis or momentum_axis as parameters, the program tries to infer them from the filename.
