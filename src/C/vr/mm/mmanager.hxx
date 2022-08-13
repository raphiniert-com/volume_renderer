#ifndef __MMANAGER_HXX__
#define __MMANAGER_HXX__
#include <vr/volumeRender.h>
#include <vr/mm/class_handle.hpp>

namespace vr {

namespace mm {

enum VolumeType {
    emission   = 0, 
    absorption = 1, 
    reflection = 2, 
    dx = 3, 
    dy = 4, 
    dz = 5
};

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

    // member functions
    MManager() {}
    ~MManager() {
        cudaDeviceReset();
    }

    void sync() {
        syncWithDevice(this->volumeEmission, this->volumeAbsorption, 
            this->volumeReflection, this->timeLastMemSync);
        mexPrintf("last sync: %u\n", this->timeLastMemSync);
        
        // initCuda(this->volumeEmission, this->volumeAbsorption, this->volumeReflection);

        if (this->volumeDx.last_update != 0 && this->volumeDy.last_update != 0 && 
            this->volumeDz.last_update != 0) {
            setGradientTextures(this->volumeDx, this->volumeDy, this->volumeDz);
        }
    }

    void resetGradients() {
        freeCudaGradientBuffers();

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
            << "\n"
            << "\t\tAbsorption (MB): " << this->bytesToMB(this->volumeAbsorption.memory_size)
            << "\n"
            << "\t\tReflection (MB): " << this->bytesToMB(this->volumeReflection.memory_size)
            << "\n"
            << "\t\tdX (MB): " << this->bytesToMB(this->volumeDx.memory_size)
            << "\n"
            << "\t\tdX (MB): " << this->bytesToMB(this->volumeDy.memory_size)
            << "\n"
            << "\t\tdX (MB): " << this->bytesToMB(this->volumeDz.memory_size)
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