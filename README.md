# fermi_level
project for detection of fermi level using arpes data through machine learning.

For a high level understanding of how the programme works, refer to technical details.docx

To run inference through the programme,

run
python inference.py --filename <path to file> --energy_axis <energy_axis> --momentum_axis <momentum_axis>
Here energy axis and momentum_axis denote the axis in which these data are stored(x, y or z). An example of doing this is shown in quicktesting.png. 
Note that if you do not give energy_axis or momentum_axis as parameters, the program tries to infer them from the filename.


------------------------------------------------------------------------------------------------------------------------------------------------

Setting up the enviornment:-

Method which should ideally work:-

conda env create -f environment.yml 


Method which actually works:-

conda deactivate
conda create --name fermilevel python=3.7.6
conda activate fermilevel
conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch
conda install -c conda-forge opencv
pip install torchsummaryX
pip install arpys


The problem may lie in using pip and conda together in the former method or platform dependency.
