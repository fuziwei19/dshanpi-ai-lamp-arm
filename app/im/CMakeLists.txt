##
# @file CMakeLists.txt
# @brief IM 组件（feishu 链路）—— 由 TClaw/IM 裁剪而来。
#        只编译 feishu 需要的源文件：utils + bus + feishu + proxy(HTTP CONNECT/SOCKS5) + certs。
#/

set(IM_PATH ${CMAKE_CURRENT_LIST_DIR})

set(IM_SRCS
    ${IM_PATH}/im_utils.c
    ${IM_PATH}/bus/message_bus.c
    ${IM_PATH}/channels/feishu_bot.c
    ${IM_PATH}/proxy/http_proxy.c
    ${IM_PATH}/certs/tls_cert_bundle.c
    ${IM_PATH}/certs/ca_bundle_mini.c
)

set(IM_INC
    ${IM_PATH}
)

########################################
# Target Configure
########################################

target_sources(${EXAMPLE_LIB}
    PRIVATE
        ${IM_SRCS}
)

target_include_directories(${EXAMPLE_LIB}
    PRIVATE
        ${IM_INC}
)
