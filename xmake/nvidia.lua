-- File: xmake/nvidia.lua (最终修正版)

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_deps("llaisys-utils")
    
    add_rules("cuda")
    add_files("../src/device/nvidia/*.cu")

    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_cxflags("-Wno-unknown-pragmas")
    end

    add_cuflags("-gencode=arch=compute_89,code=sm_89")
    add_cuflags("-Wno-deprecated-gpu-targets")
    
    -- 【！！！最关键的修正！！！】
    -- add_rules("cuda") 会为静态库默认启用 -rdc=true (可重定位设备代码),
    -- 这个功能会强制使用静态链接并导致无法解决的 "undefined symbol" 错误。
    -- 我们在这里明确地禁用它，回到标准的、干净的编译模式。
    add_cuflags("-rdc=false", {force = true})
    
    -- 确保依赖方在最终链接时能自动链接 CUDA 运行时
    -- 将 cudart 添加为此静态目标的链接依赖，这会被依赖它的共享目标继承
    if not is_plat("windows") then
        add_linkdirs("/usr/local/cuda/lib64")
        add_links("cudart")
    end

    -- 我们什么都不加，让最终的共享库来决定链接方式
    on_install(function (target)
        import("core.project.project")
        local prj_root = project.directory()
        local obj_dir = path.join(prj_root, "build/.objs/llaisys-device-nvidia/linux/x86_64/release/src/device/nvidia")
        local build_dir = path.join(prj_root, "build/linux/x86_64/release")
        local dlink = path.join(build_dir, "nvidia_dlink.o")
        local archive = path.join(build_dir, "libllaisys-device-nvidia.a")

        if os.isdir(obj_dir) and os.exists("/usr/local/cuda/bin/nvcc") then
            os.execf("/usr/local/cuda/bin/nvcc --device-link %s/*.o -o %s -Xcompiler -fPIC", obj_dir, dlink)
            if os.exists(dlink) then
                os.execf("/usr/bin/ar r %s %s", archive, dlink)

                local llaisys_objs_dir = path.join(prj_root, "build/.objs/llaisys/linux/x86_64/release/src/llaisys")
                local llaisys_objs = os.files(path.join(llaisys_objs_dir, "*.o"))
                local archives = {
                    path.join(build_dir, "libllaisys-ops.a"),
                    path.join(build_dir, "libllaisys-ops-cpu.a"),
                    path.join(build_dir, "libllaisys-tensor.a"),
                    path.join(build_dir, "libllaisys-core.a"),
                    path.join(build_dir, "libllaisys-device.a"),
                    path.join(build_dir, "libllaisys-device-nvidia.a"),
                    path.join(build_dir, "libllaisys-utils.a"),
                    path.join(build_dir, "libllaisys-device-cpu.a")
                }
                local objs_str = ""
                if llaisys_objs then
                    objs_str = table.concat(llaisys_objs, " ")
                end
                local cmd = "/usr/bin/g++ -shared -m64 -fPIC " .. objs_str .. " " .. dlink .. " " .. table.concat(archives, " ") ..
                            " -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -s -lcudart -lcudadevrt -lrt -lpthread -ldl -o " .. path.join(build_dir, "libllaisys.so")
                os.exec(cmd)

                local dest = path.join(prj_root, "python/llaisys/libllaisys/libllaisys.so")
                if os.exists(path.join(build_dir, "libllaisys.so")) then
                    os.cp(path.join(build_dir, "libllaisys.so"), dest)
                end
            end
        end
    end)
target_end()

target("llaisys-device")
    add_deps("llaisys-device-nvidia")
target_end()