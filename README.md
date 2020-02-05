# Volume Renderer <small>for use with MATLAB®</small>

_Volume Renderer <small>for use with MATLAB®</small>_ extends MATLAB® by a GPU-accelerated volume render command that handles 3D volumetric data. The core application is implemented in C/C++. To guarantee fast computations the render process computes on the GPU. This is realized by NVIDIA® CUDA®. Furthermore _Volume Renderer <small>for use with MATLAB®</small>_ provides the following features:

*   **Special memory management:** Due to restricted GPU memory and the requirement to render more than one volume in one scene, we developed a special memory management to enable the rendering of huge data sets in separate rendering passes. Afterward these separately rendered images are combined to one image using MATLAB®.
*   **Generic illumination model:** We developed a generic illumination model that is easy to extend with other illumination functions. The provided function is the Henyey-Greenstein phase function.
*   **Stereo rendering:** In some use cases there is a nice feature to work with stereo images. Thus, the renderer offers the possibility to render off-axis stereo images.
*   **High usability:** To enable a high usability a MATLAB® interface consisting of several MATLAB® classes has been developed. Due to this interface it is uncomplicated to generate movies.


## Requirements
*   CUDA® capable NVIDIA® graphics device with at least [Kepler™](https://en.wikipedia.org/wiki/Kepler_(microarchitecture)) architecture
*   Linux computer (64 bit) with installed NVIDIA® driver and CUDA®
*   MATLAB® <sup id="a1">[1](#f1)</sup> with [Parallel Computing Tookbox](https://mathworks.com/products/parallel-computing.html) for compilation (requires [mexcuda](https://de.mathworks.com/help/parallel-computing/mexcuda.html))


## Installation
First download and extract or clone the repository.
Next, enter the local folder with the render code in matlab. Run the `make.m` file inside `src`. This command will compile all mex-files for the renderer.
Either, enter `src/matlab` to run code and place your matlab renderer code there, or setup matlab to load this folder at each startup into its search path as described [here](https://de.mathworks.com/help/matlab/matlab_env/add-folders-to-matlab-search-path-at-startup.html) (recommended by us).


## Example
The following video demonstrates the power of the renderer<sup id="a2">[2](#f2)</sup>:

![Demo CountPages alpha](docs/example_vr_zebra.gif)


## License
- This work is licensed under [GNU Affero General Public License version 3](https://opensource.org/licenses/AGPL-3.0). 
- Copyright 2020 © [Raphael Scheible](raphiniert.com)

## Acknowledgments
_Volume Renderer for use with MATLAB®_ was originally developed as a student project by Raphael Scheible at University of Freiburg supervised by [Benjamin Ummenhofer](http://lmb.informatik.uni-freiburg.de/people/ummenhof/) and [apl. Prof. Dr. Olaf Ronneberger](http://lmb.informatik.uni-freiburg.de/people/ronneber/).

## References
[1]  <a id="ref1"></a>Ronneberger, O and Liu, K and Rath, M and Ruess, D and Mueller, T and Skibbe, H and Drayer, B and Schmidt, T and Filippi, A and Nitschke, R and Brox, T and Burkhardt, H and Driever, W. **[ViBE-Z: A Framework for 3D Virtual Colocalization Analysis in Zebrafish Larval Brains](http://lmb.informatik.uni-freiburg.de//Publications/2012/RLSDSBB12) .** 2012. _Nature Methods,_ 9(7):735--742. [↩](#r1)

---

<a id="f1"></a>1: tested and developed under R2019b; might work from R2015b [↩](#a1)  
<a id="f2"></a>2: provided by Benjamin Ummenhofer, data from <a id="r1">[[1]](#ref1)</a> [↩](#a2)
