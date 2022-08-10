#include "mex.h"
#include "class_handle.hpp"
#include <string>

// The class that we are interfacing to
class MManager
{
public:
    MManager() { this->_counter=0; }
    ~MManager() { mexPrintf("Calling destructor\ncounter=%d\n", this->_counter); }
    void inc() { this->_counter++; };
    void dec() { this->_counter--; };
    void set(const int value) { this->_counter = value; };
    void print() { mexPrintf("current value: %d\n", this->_counter); }
private:
    unsigned int _counter;
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
    // Get the command string
    char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");
        
    // New
    if (!strcmp("new", cmd)) {
        // Check parameters
        if (nlhs != 1)
            mexErrMsgTxt("New: One output expected.");
        // Return a handle to a new C++ instance
        plhs[0] = convertPtr2Mat<MManager>(new MManager);
        return;
    }
    
    // Check there is a second input, which should be the class instance handle
    if (nrhs < 2)
		mexErrMsgTxt("Second input should be a class instance handle.");
    
    // Delete
    if (!strcmp("delete", cmd)) {
        // Destroy the C++ object
        destroyObject<MManager>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        return;
    }
    
    // Get the class instance pointer from the second input
    MManager* mmanager_instance = convertMat2Ptr<MManager>(prhs[1]);
    
    // Call the various class methods
    // dec
    if (!strcmp("dec", cmd)) {
        // Check parameters
        if (nlhs < 0 || nrhs < 2)
            mexErrMsgTxt("dec: Unexpected arguments.");
        // Call the method
        mmanager_instance->dec();
        return;
    }
    // inc
    if (!strcmp("inc", cmd)) {
        // Check parameters
        if (nlhs < 0 || nrhs < 2)
            mexErrMsgTxt("inc: Unexpected arguments.");
        // Call the method
        mmanager_instance->inc();
        return;
    }
    // print
    if (!strcmp("print", cmd)) {
        // Check parameters
        if (nlhs < 0 || nrhs < 2)
            mexErrMsgTxt("print: Unexpected arguments.");
        // Call the method
        mmanager_instance->print();
        return;
    }

    if (!strcmp("set", cmd)) {
        // Check parameters
        if (nlhs < 0 || nrhs < 3)
            mexErrMsgTxt("dec: Unexpected arguments.");
        // Call the method
        mmanager_instance->set((unsigned int)mxGetScalar(prhs[2]));
        return;
    }

    if (!strcmp("getAddress", cmd)) {
        mexPrintf("%d", nrhs);
        // Check parameters
        if (nlhs < 0 || nrhs < 3)
            mexErrMsgTxt("getAdress: Unexpected arguments.");
        // Call the method
        mexPrintf("memory: %d", prhs[2]);
        return;
    }
    
    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized.");
}
