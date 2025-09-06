# -fvisibility=hidden can not hidden those symbols, may be declaration with extern C or visibility sets to default.
# bthread_jump_fcontext
# bthread_make_fcontext
# _ZN6google8protobuf8internal8byteswapILi1EEEvPv
# _ZN6google8protobuf8internal8byteswapILi4EEEvPv
# _ZN6google8protobuf8internal8byteswapILi8EEEvPv
{
    global:
        dlsym;
        aclnnMatmulGetWorkspaceSize;
        aclnnMatmul;
        aclnnGroupedMatmulV2GetWorkspaceSize;
        aclnnGroupedMatmulV2;
        HcclAllReduce;
        HcclBroadcast;
        HcclReduceScatter;
        HcclReduce;
        HcclAlltoAll;
        HcclAlltoAllV;
        HcclAllGather;
        HcclBarrier;
        HcclScatter;
        HcclSend;
        HcclRecv;
        HcclBatchSendRecv;
    local:
        *;
};
