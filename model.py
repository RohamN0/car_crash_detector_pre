import torchvision.transforms.functional as F
import torch.nn.functional as FF
import pandas as pd
import numpy as np
import cv2, torch, os

class VideoPipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.frame_interval = 6
        self.video_matrix_output_dir = './videos/video_matrices'
        self.mask_matrix_output_dir = './videos/masked_video_matrices'

        self.df = pd.read_csv('./data/train.csv')

        self.id = video_path.split('/')[-1].replace('.mp4', '')
        
        self.df = self.df[self.df['id'] == int(self.id)].iloc[0]
        
        self.time_of_event = float(self.df['time_of_event'])
        self.target = int(self.df['target'])

    def resize_on_gpu(self, frames):
        frames_tensor = torch.from_numpy(frames).permute(0,3,1,2).float().to(self.device)
        resized = F.resize(frames_tensor, [224,224])

        return resized.permute(0,2,3,1).cpu().numpy()
    
    def pad_or_truncate(self, seq, max_len=10):
        seq = np.array(seq)
        if len(seq) > max_len:
            return seq[:max_len]

        if len(seq) < max_len:
            last = seq[-1]
            padding = np.repeat(last[None, ...], max_len - len(seq), axis=0)
            return np.concatenate([seq, padding], axis=0)

        return seq

    def get_target(self) :
        return self.target

    def video_to_matrix(self):
        os.makedirs(self.video_matrix_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        if self.target == 1:
            start_time = max(self.time_of_event - 2, 0)
            end_time = self.time_of_event
        else:
            start_time = np.random.uniform(0, duration - 2)
            end_time = start_time + 2

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        frame_idx = start_frame

        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok: break

            if frame_idx % self.frame_interval == 0:
                frames.append(frame)

            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames extracted!")

        frames = np.stack(frames)
        frames = self.resize_on_gpu(frames)
        frames = self.pad_or_truncate(frames)

        self.video_matrix_path = f"{self.video_matrix_output_dir}/video_{self.id}.npy"
        np.save(self.video_matrix_path, frames)

        return self.video_matrix_path

    def masked_video_to_matrix(self):
        os.makedirs(self.mask_matrix_output_dir, exist_ok=True)

        frames = np.load(self.video_matrix_path)  # (N,H,W,3)
        frames_tensor = torch.from_numpy(frames).permute(0,3,1,2).float().to(self.device)/255.0

        N = frames_tensor.shape[0]
        gray = (0.2989*frames_tensor[:,0] + 0.5870*frames_tensor[:,1] + 0.1140*frames_tensor[:,2]).unsqueeze(1)

        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=self.device).view(1,1,3,3)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=self.device).view(1,1,3,3)
        kernel = torch.ones((1,1,3,3), device=self.device)

        mask_frames = []

        for i in range(1, N):
            prev = gray[i-1:i]
            next = gray[i:i+1]

            Ix = FF.conv2d(prev, sobel_x, padding=1)
            Iy = FF.conv2d(prev, sobel_y, padding=1)
            It = next - prev

            Ixx = FF.conv2d(Ix*Ix, kernel, padding=1)
            Iyy = FF.conv2d(Iy*Iy, kernel, padding=1)
            Ixy = FF.conv2d(Ix*Iy, kernel, padding=1)
            Ixt = FF.conv2d(Ix*It, kernel, padding=1)
            Iyt = FF.conv2d(Iy*It, kernel, padding=1)

            det = Ixx*Iyy - Ixy*Ixy + 1e-6
            u = (Iyy*(-Ixt) - Ixy*(-Iyt)) / det
            v = (-Ixy*(-Ixt) + Ixx*(-Iyt)) / det

            u = u[0,0]
            v = v[0,0]

            magnitude = torch.sqrt(u*u + v*v)
            magnitude = 255*(magnitude - magnitude.min())/(magnitude.max() - magnitude.min() + 1e-6)

            angle = torch.atan2(v,u)*180/np.pi/2

            dy_u, dx_u = torch.gradient(u)
            dy_v, dx_v = torch.gradient(v)
            divergence = dx_u + dy_v
            divergence = ((torch.tanh(divergence)+1)/2)*255

            stacked = torch.stack([magnitude, angle, divergence], dim=0).clamp(0,255).byte()
            mask_frames.append(stacked.cpu())

        mask_frames = torch.stack(mask_frames).numpy()
        mask_frames = self.pad_or_truncate(mask_frames)

        out_path = f'{self.mask_matrix_output_dir}/masked_video_{self.id}.npy'
        np.save(out_path, mask_frames)

        return out_path

    def process(self):
        video_np = self.video_to_matrix()
        mask_np = self.masked_video_to_matrix()
        
        return video_np, mask_np
