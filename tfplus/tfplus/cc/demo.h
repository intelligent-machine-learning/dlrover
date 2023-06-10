// Copyright 2023 tfplus.

#ifndef _V_TFPLUS_CC_DEMO_H_
#define _V_TFPLUS_CC_DEMO_H_

namespace tfplus {

class Demo {
 public:
    void print_localtime();
};

}

extern "C" {
    void print_localtime() {
        tfplus::Demo dm;
        dm.print_localtime();
    }
}

#endif  // _V_TFPLUS_CC_DEMO_H_
