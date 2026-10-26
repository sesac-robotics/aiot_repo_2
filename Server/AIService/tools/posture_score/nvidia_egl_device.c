/** NVIDIA EGL 장치를 선택하는 프로세스 전용 검사 어댑터.
 * 기본 디스플레이 요청만 NVIDIA 장치 디스플레이로 연결한다.
 * 초기화 실패 시 NULL을 반환하며 시스템 드라이버 설정은 변경하지 않는다.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>
typedef void* EGLDisplay;
typedef void* EGLDeviceEXT;
typedef unsigned int EGLBoolean;
typedef int EGLint;
/**
 * @brief 기본 디스플레이 요청을 NVIDIA EGL 장치로 연결한다.
 * @param native_display EGL 기본 디스플레이 NULL 또는 명시한 네이티브 핸들.
 * @return 선택한 장치의 디스플레이 핸들. 단위가 없는 불투명 포인터이다.
 * @retval NULL 라이브러리·확장 함수·NVIDIA 장치가 없거나 조회에 실패한 경우.
 * @note 명시한 네이티브 디스플레이는 원래 EGL 함수에 전달한다.
 */
EGLDisplay eglGetDisplay(void* native_display) {
  void* lib = dlopen("libEGL.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (!lib) return NULL;
  if (native_display != NULL) {
    EGLDisplay (*original)(void*) = dlsym(lib, "eglGetDisplay");
    return original ? original(native_display) : NULL;
  }
  void* (*getproc)(const char*) = dlsym(lib, "eglGetProcAddress");
  if (!getproc) return NULL;
  EGLBoolean (*query)(EGLint, EGLDeviceEXT*, EGLint*) =
      getproc("eglQueryDevicesEXT");
  const char* (*string)(EGLDeviceEXT, EGLint) =
      getproc("eglQueryDeviceStringEXT");
  EGLDisplay (*platform)(unsigned int, void*, const EGLint*) =
      getproc("eglGetPlatformDisplayEXT");
  if (!query || !string || !platform) return NULL;
  EGLDeviceEXT devices[16];
  EGLint count = 0;
  if (!query(16, devices, &count)) return NULL;
  for (int i = 0; i < count; i++) {
    const char* extensions = string(devices[i], 0x3055);
    if (extensions && strstr(extensions, "EGL_NV_device_cuda"))
      return platform(0x313F, devices[i], NULL);
  }
  return NULL;
}
