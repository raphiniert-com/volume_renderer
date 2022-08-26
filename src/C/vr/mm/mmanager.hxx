#ifndef __MMANAGER_HXX__
#define __MMANAGER_HXX__
#include <vr/volumeRender.h>
#include <vr/mm/class_handle.hpp>

namespace vr {

namespace mm {

class MManager
{
public:
    // member variables
    uint64_t timeLastMemSync = 0;

    Volume& volumeEmission = make_volume(NULL, 0, make_cudaExtent(0,0,0));
    Volume& volumeAbsorption = make_volume(NULL, 0, make_cudaExtent(0,0,0));
    Volume& volumeReflection = make_volume(NULL, 0, make_cudaExtent(0,0,0));

    Volume& volumeDx = make_volume(NULL, 0, make_cudaExtent(0,0,0));
    Volume& volumeDy = make_volume(NULL, 0, make_cudaExtent(0,0,0));
    Volume& volumeDz = make_volume(NULL, 0, make_cudaExtent(0,0,0));

    Volume& volumeLight = make_volume(NULL, 0, make_cudaExtent(0,0,0));

    // pointer to device addresses
    cudaArray * ptr_d_volumeEmission = 0;
    cudaArray * ptr_d_volumeAbsorption = 0;
    cudaArray * ptr_d_volumeReflection = 0;

    cudaArray * ptr_d_volumeDx = 0;
    cudaArray * ptr_d_volumeDy = 0;
    cudaArray * ptr_d_volumeDz = 0;
    
    cudaArray * ptr_d_volumeLight = 0;

    // think about memory swapping mechanism here: if v_em_old == v_em_new, and some other stuff is equivalent, just swap here without new assignment.

    // member functions
    MManager() {}
    ~MManager() {
        cudaDeviceReset();
    }

    size_t getRequiredMemory() {
        size_t requiredRAM = 0;
        // // emission is required in anycase
        requiredRAM += this->volumeEmission.memory_size;

        // check if absorption is unique
        if (this->volumeEmission != this->volumeAbsorption &&
            this->volumeReflection != this->volumeAbsorption) {
          requiredRAM += this->volumeAbsorption.memory_size;
        }

        // check if reflection is unique
        if (this->volumeEmission != this->volumeReflection &&
           this-> volumeReflection != this->volumeAbsorption) {
          requiredRAM += this->volumeReflection.memory_size;
        }

        // if gradients are passed through
        requiredRAM += this->volumeDx.memory_size + 
                       this->volumeDy.memory_size + 
                       this->volumeDz.memory_size;

        return requiredRAM;
        // check if there is enough free VRam
        // if not program will stop with an error msg
        // checkFreeDeviceMemory(requiredRAM);
    }

    /*! \fn void checkFreeDeviceMemory(size_t aRequiredRAMInBytes)
      * 	\brief checks if there is enough free device memory available
      *  \param aRequiredRAMInBytes required memory in bytes
      *
      * If there is not enough free device memory available the program will be
      * stopped and an error message will be displayed in the matlab interface. The
      * user will be informed how much memory he wanted to allocate and how much
      * 	(free) memory the device offers.
      */
    static void checkFreeDeviceMemory(size_t aRequiredRAMInBytes) {
      size_t totalMemoryInBytes, curAvailMemoryInBytes;

      bool isEnough = false;
      cudaMemGetInfo(&curAvailMemoryInBytes, &totalMemoryInBytes);
#ifdef DEBUG

  mexPrintf(
    "\ttotal memory: %ld MB, free memory: %ld MB, required memory: %ld MB\n",
    totalMemoryInBytes / (1024 * 1024), curAvailMemoryInBytes / (1024 * 1024),
    aRequiredRAMInBytes / (1024 * 1024));

#endif

      isEnough = (curAvailMemoryInBytes >= aRequiredRAMInBytes);
      // cuCtxDetach(context); // Destroy context

      if (!isEnough) {
          std::ostringstream os;
          os << "insufficient free VRAM!\n"
          << "\tTotal Memory (MB): \t" << totalMemoryInBytes / (1024 * 1024)
          << "\n"
          << "\tFree Memory (MB): \t" << curAvailMemoryInBytes / (1024 * 1024)
          << "\n"
          << "\tRequired memory (MB): \t" << aRequiredRAMInBytes / (1024 * 1024)
          << "\n";

          mexErrMsgTxt(os.str().c_str());
      }
    }

    void sync() {
      // reset cuda memory of gradient volumes, if not set anymore
      if ((this->volumeDx.last_update == 0 && this->ptr_d_volumeDx != 0) ||
          (this->volumeDy.last_update == 0 && this->ptr_d_volumeDy != 0) ||
          (this->volumeDz.last_update == 0 && this->ptr_d_volumeDz != 0)) {
            resetGradients();
          }

      syncWithDevice(
          this->volumeEmission, this->volumeAbsorption,
          this->volumeReflection, this->timeLastMemSync,
          this->ptr_d_volumeEmission,
          this->ptr_d_volumeAbsorption,
          this->ptr_d_volumeReflection);
      mexPrintf("last sync: %u\n", this->timeLastMemSync);

      if (this->volumeDx.last_update != 0 && this->volumeDy.last_update != 0 && 
          this->volumeDz.last_update != 0) {
          setGradientTextures(
            this->volumeDx, this->volumeDy, this->volumeDz,
            this->ptr_d_volumeDx, this->ptr_d_volumeDy, this->ptr_d_volumeDz,
            this->timeLastMemSync
          );
      }
    }

    void resetGradients() {
      if (this->ptr_d_volumeDx == 0 || this->ptr_d_volumeDy == 0 || this->ptr_d_volumeDz == 0)
        freeCudaGradientBuffers(this->ptr_d_volumeDx, this->ptr_d_volumeDy, this->ptr_d_volumeDz);

      volumeDx = make_volume(NULL, 0, make_cudaExtent(0,0,0));
      volumeDy = make_volume(NULL, 0, make_cudaExtent(0,0,0));
      volumeDz = make_volume(NULL, 0, make_cudaExtent(0,0,0));
    }

    std::string memInfo() {
        size_t totalMemoryInBytes, curAvailMemoryInBytes;
        cudaMemGetInfo(&curAvailMemoryInBytes, &totalMemoryInBytes);

        bool similarityEmAb = (this->volumeEmission == this->volumeAbsorption);
        bool similarityEmRe = (this->volumeEmission == this->volumeReflection);
        bool similarityAbRe = (this->volumeAbsorption == this->volumeReflection);

        std::ostringstream os;
        os  << "Memory Information\n"
            << "------------------------------"
            << "\n"
            << "\tGPU"
            << "\n"
            << "\t---"
            << "\n"
            << "\t\tTotal Memory (MB): \t" << this->bytesToMB(totalMemoryInBytes)
            << "\n"
            << "\t\tFree Memory (MB): \t" << this->bytesToMB(curAvailMemoryInBytes)
            << "\n"
            << "\t\tUsed Memory (MB): \t" << this->bytesToMB(totalMemoryInBytes - curAvailMemoryInBytes)
            << "\n"
            << "\n"
            << "\t\tVolumes"
            << "\n"
            << "\t\t-------"
            << "\n"
            << "\t\tEmission (MB): " << this->bytesToMB(this->volumeEmission.memory_size)
            << " ptr: " << this->ptr_d_volumeEmission
            << "\n"
            << "\t\tAbsorption (MB): " << this->bytesToMB(this->volumeAbsorption.memory_size)
            << " ptr: " << this->ptr_d_volumeAbsorption
            << "\n"
            << "\t\tReflection (MB): " << this->bytesToMB(this->volumeReflection.memory_size)
            << " ptr: " << this->ptr_d_volumeReflection
            << "\n"
            << "\t\tdX (MB): " << this->bytesToMB(this->volumeDx.memory_size)
            << " ptr: " << this->ptr_d_volumeDx
            << "\n"
            << "\t\tdX (MB): " << this->bytesToMB(this->volumeDy.memory_size)
            << " ptr: " << this->ptr_d_volumeDy
            << "\n"
            << "\t\tdX (MB): " << this->bytesToMB(this->volumeDz.memory_size)
            << " ptr: " << this->ptr_d_volumeDz
            << "\n"
            << "\t\tlight (MB): " << this->bytesToMB(this->volumeLight.memory_size)
            << " ptr: " << this->ptr_d_volumeLight
            << "\n"
            << "\n"
            << "\t\tSimilarity of Volumes"
            << "\n"
            << "\t\t---------------------"
            << "\n"
            << "\t\t\tEm\tAb\tRe"
            << "\n"
            << "\t\tEm\t" << 1 << "\t"
            << "\n"
            << "\t\tAb\t" << similarityEmAb << "\t" << 1
            << "\n"
            << "\t\tRe\t" << similarityEmRe << "\t" << similarityAbRe << "\t" << 1
            << "\n"
            << "\n";

        return os.str();
    }

private:
    double bytesToMB(size_t bytes) {
        return (double)bytes/(1024.0 * 1024.0);
    }
}; // class
}; // namespace mm
}; // namespace vr
#endif // __MMANAGER_HXX__