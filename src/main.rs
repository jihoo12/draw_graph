use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

// 정육면체 정점 데이터 (8개만 정의)
const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.5, -0.5, 0.5],
        color: [1.0, 0.0, 0.0],
    }, // 0
    Vertex {
        position: [0.5, -0.5, 0.5],
        color: [0.0, 1.0, 0.0],
    }, // 1
    Vertex {
        position: [0.5, 0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    }, // 2
    Vertex {
        position: [-0.5, 0.5, 0.5],
        color: [1.0, 1.0, 0.0],
    }, // 3
    Vertex {
        position: [-0.5, -0.5, -0.5],
        color: [1.0, 0.0, 1.0],
    }, // 4
    Vertex {
        position: [0.5, -0.5, -0.5],
        color: [0.0, 1.0, 1.0],
    }, // 5
    Vertex {
        position: [0.5, 0.5, -0.5],
        color: [1.0, 1.0, 1.0],
    }, // 6
    Vertex {
        position: [-0.5, 0.5, -0.5],
        color: [0.0, 0.0, 0.0],
    }, // 7
];

// 정점을 연결할 순서 (인덱스)
const INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0, // 앞면
    1, 5, 6, 6, 2, 1, // 오른쪽
    5, 4, 7, 7, 6, 5, // 뒷면
    4, 0, 3, 3, 7, 4, // 왼쪽
    3, 2, 6, 6, 7, 3, // 윗면
    4, 5, 1, 1, 0, 4, // 아랫면
];

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: Mat4,
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    window: Arc<Window>,
    rotation: f32,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    depth_texture_view: wgpu::TextureView,
    camera_radius: f32,
    is_dragging: bool,
    yaw: f32,
    pitch: f32,
}

impl State {
    async fn new(window: Arc<Window>) -> State {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        let size = window.inner_size();
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float, // 깊이 값을 저장할 포맷
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- 카메라 설정 ---
        let aspect = config.width as f32 / config.height as f32;
        let proj = Mat4::perspective_rh(f32::to_radians(45.0), aspect, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 1.0, 2.0), Vec3::ZERO, Vec3::Y);
        let camera_uniform = CameraUniform {
            view_proj: proj * view,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 셰이더와 버퍼를 연결하는 레이아웃
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // 더 가까운 것만 그리기
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            camera_buffer,
            camera_bind_group,
            rotation: 0.0,
            index_buffer,
            num_indices,
            depth_texture_view,
            camera_radius: 4.0,
            is_dragging: false,
            yaw: f32::to_radians(-90.0),
            pitch: f32::to_radians(20.0)
        }
    }
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // 1. 카메라 업데이트 로직 (기존과 동일)
        let aspect = self.config.width as f32 / self.config.height as f32;
        let proj = Mat4::perspective_rh(f32::to_radians(45.0), aspect, 0.1, 100.0);
        let x = self.camera_radius * self.pitch.cos() * self.yaw.cos();
        let y = self.camera_radius * self.pitch.sin();
        let z = self.camera_radius * self.pitch.cos() * self.yaw.sin();

        let view = Mat4::look_at_rh(
            Vec3::new(x, y, z),
            Vec3::ZERO,
            Vec3::Y
        );

        let camera_uniform = CameraUniform {
            view_proj: proj * view,
        };

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // ★ 깊이 스텐실 첨부 추가 ★
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view, // 미리 생성해둔 뷰
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // 가장 먼 거리로 초기화
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16); //
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1); //
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
    // 2. 렌더링 로직
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            // 창 크기가 바뀌면 투영 행렬도 다시 계산해서 업데이트해야 합니다.
            let aspect = self.config.width as f32 / self.config.height as f32;
            let proj = Mat4::perspective_rh(f32::to_radians(45.0), aspect, 0.1, 100.0);
            let view = Mat4::look_at_rh(Vec3::new(0.0, 1.0, 2.0), Vec3::ZERO, Vec3::Y);
            let camera_uniform = CameraUniform {
                view_proj: proj * view,
            };
            self.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[camera_uniform]),
            );
        }
    }
}

// App 구조체와 main 함수는 이전과 거의 동일하므로 생략하거나 기존 코드를 사용하세요.
struct App {
    state: Option<State>,
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap(),
            );
            self.state = Some(pollster::block_on(State::new(window)));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = match self.state.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::MouseInput { state: button_state, button: winit::event::MouseButton::Left, .. } => {
                state.is_dragging = button_state == winit::event::ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                // 실제로는 이전 좌표와의 차이(Delta)를 써야 하지만,
                // 간단히 구현하기 위해 좌표의 X값을 회전에 직접 연결해 볼 수 있습니다.
                if state.is_dragging {
                    // 마우스 X 좌표의 변화에 따라 rotation 변경 (0.01은 감도 조절용)
                    state.yaw += 0.005;
                    state.pitch = (state.pitch + 0.005).clamp(f32::to_radians(-89.0), f32::to_radians(89.0));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_amount = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                // 휠을 밀면 가까워지고(-), 당기면 멀어지게(+) 설정
                // 최소 거리 1.0, 최대 거리 20.0으로 제한(clamp)
                state.camera_radius = (state.camera_radius - scroll_amount).clamp(1.0, 20.0);
            }
            WindowEvent::RedrawRequested => {
                // 화면을 그리고
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(e) => eprintln!("{:?}", e),
                }
                // ★ 중요: 그리기가 끝나자마자 다음 프레임을 즉시 요청합니다.
                state.window.request_redraw();
            }
            _ => (),
        }
    }

    // 이 부분은 비워두거나 제거해도 RedrawRequested 내의 호출이 우선됩니다.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { state: None };
    event_loop.run_app(&mut app).unwrap();
}
