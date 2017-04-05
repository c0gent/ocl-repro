extern crate ocl;
use ocl::*;
use ocl::enums::*;
use ocl::builders::*;

// The main purpose of this program is to take a long time to compute.
const PROGRAM: &'static str = r##"
__kernel void kern(__global float* const res)
{
    ulong const idx = get_global_id(0);
    ulong v = 0;
    for (ulong i = 0; i < idx * idx; i++) {
        v += i;
        // v *= 2;
    }
    res[idx] = (float) v;
}
"##;

fn main() {
    let (platform, device) = first_gpu();
    let context = Context::builder().devices(device).platform(platform).build().unwrap();
    let program = ProgramBuilder::new()
        .src(PROGRAM)
        .devices(device)
        .cmplr_opt("-cl-denorms-are-zero -cl-fast-relaxed-math")
        .build(&context).unwrap();
    let queue = Queue::new(&context, device, Some(flags::QUEUE_PROFILING_ENABLE)).unwrap();
    let work_size = 250000;
    let buff = Buffer::<f32>::builder()
        .queue(queue.clone())
        .dims(work_size)
        .build().unwrap();
    let kernel = Kernel::new("kern", &program).unwrap()
        .queue(queue.clone())
        .gws(work_size)
        .arg_buf(&buff);
    
    kernel.enq().unwrap();
    queue.finish().unwrap();

    let mut out = vec![2.0; work_size];
    buff.read(&mut out).enq().unwrap();
    println!("Result count: {}", out.len());
}

pub fn first_gpu() -> (Platform, Device) {
    let mut out = vec![];
    for plat in Platform::list() {
        if let Ok(all_devices) = Device::list_all(&plat) {
            for dev in all_devices {
                out.push((plat.clone(), dev));
            }
        }
    }

    // Prefer GPU
    out.sort_by(|&(_, ref a), &(_, ref b)| {
        let a_type = a.info(DeviceInfo::Type);
        let b_type = b.info(DeviceInfo::Type);
        if let (DeviceInfoResult::Type(a_type), DeviceInfoResult::Type(b_type)) = (a_type, b_type) {
            b_type.cmp(&a_type)
        } else {
            (0).cmp(&0)
        }
    });

    out.first().unwrap().clone()
}
