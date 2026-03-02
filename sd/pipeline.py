import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt: str, 
    uncond_prompt: str, 
    input_image = None, 
    strength = 0.8, # img2img에서 output이 얼마나 input과 닮게할지 결정하는 parameter
    do_cfg = True, 
    cfg_scale = 7.5, 
    sampler_name = "ddpm",  
    n_inference_steps=50, 
    models={},
    seed=None,
    device=None, 
    idle_device=None, 
    tokenizer=None
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        # idle device로 옮겨주는 헬퍼
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device) # 난수 랜덤 생성기

        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip = clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids           
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding = "max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            
            context = clip(tokens)

        to_idle(clip) # 이제 clip CPU로 내리기

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        # img2img case
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1)) # pixel 범위 0~255를 vae입력에 맞추어 -1 ~ 1로 변환
            input_image_tensor = input_image_tensor.unsqueeze(0) # batch 차원 추가
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        # text2img case
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            # cfg이면, cond/uncond를 한 번에 처리하기 위해 batch를 2배로 늘림
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)
            
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp = True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

# 값 범위를 old_range에서 new_range로 선형적으로 변환하는 함수
def rescale(x, old_range, new_range, clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp: # 범위를 벗어난 값을 범위 내부로 맞춰줌
        x = x.clamp(new_min, new_max)
    return x

# timestep t(정수)를 sin cos 기반 embedding vector로 바꿔줌
def get_time_embedding(timestep):
    # 길이 160의 vector를 만들고 10000^(-index/160)을 계산한다.
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    return torch.cat([torch.cos(x), torch.sin(x)], dim = -1)
    


                
                


 


