SUMMARY = "Hand gesture (left/right) recognition with Pi camera and TFLite C API"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://gesture.c \
    file://gesture.service \
    file://model.tflite \
    file://labels.txt \
"
S = "${WORKDIR}"

DEPENDS = "libtensorflow-lite-c"
RDEPENDS:${PN} = "libtensorflow-lite-c libcamera-apps"

inherit systemd
SYSTEMD_SERVICE:${PN} = "gesture.service"

do_compile() {
    ${CC} ${CFLAGS} ${LDFLAGS} -o gesture gesture.c -ltensorflowlite_c
}

do_install() {
    install -D -m 0755 gesture ${D}${bindir}/gesture
    install -D -m 0644 model.tflite ${D}${datadir}/gesture/model.tflite
    install -D -m 0644 labels.txt ${D}${datadir}/gesture/labels.txt
    install -D -m 0644 gesture.service ${D}${systemd_system_unitdir}/gesture.service
}
