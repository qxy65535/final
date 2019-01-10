#### - [执行指令](#shell)

<h2 id="shell">执行指令</h2>

**compile**

```shell
$ make
```

**encode**

```shell
$ ./c63enc -w 352 -h 288 -o tmp/FOREMAN_352x288_30_orig_01.c63 video/FOREMAN_352x288_30_orig_01.yuv
$ ./c63enc -w 1920 -h 1080 -o tmp/1080p_tractor.c63 video/1080p_tractor.yuv
```

**decode**
```shell
$ ./c63dec tmp/FOREMAN_352x288_30_orig_01.c63  tmp/foreman.yuv
$ ./c63dec tmp/1080p_tractor.c63  tmp/tractor.yuv
```

**play the raw yuv file**

```shell
$ vlc --rawvid-width 352 --rawvid-height 288 --rawvid-fps 30 --rawvid-chroma I420 tmp/foreman.yuv
$ vlc --rawvid-width 1920 --rawvid-height 1080 --rawvid-fps 30 --rawvid-chroma I420 tmp/tractor.yuv
```

**time account**
```shell
$ gprof c63enc gmon.out -p
```