%% 第一课：圆形光瞳 -> 离焦 PSF -> 模糊图 -> 维纳复原
% 本脚本只依赖 MATLAB 基础功能，无需图像处理工具箱。
% 模型：单色、标量、非相干、空间不变、点采样探测器。
% FFT 采用周期边界；w 是瞳边缘相对中心的光程差/波长，不是毫米位移。
clear; close all; clc;

%% 1. 采样和光学参数；长度单位统一为微米
N = 256;
pixel_pitch_um = 2.0;
wavelength_um = 0.55;
f_number = 8.0;
defocus_waves = 0.8;
noise_std = 0.005;
balance = 0.01;  % 正则参数，不能简单等同于本实验的噪声方差
assert(pixel_pitch_um <= wavelength_um*f_number/2, ...
    '像素间距过大，超过非相干光学传递函数的 Nyquist 采样限制');

%% 2. 生成线性强度测试图；不是物体的相位分布
[X,Y] = meshgrid((0:N-1)/N, (0:N-1)/N);
original = 0.18 + 0.12*X;
original((X-0.28).^2+(Y-0.32).^2 < 0.12^2) = 0.9;
original(abs(X-0.67)<0.14 & abs(Y-0.32)<0.1) = 0.65;
original(abs(X-0.45)<0.22 & abs(Y-0.72)<0.075) = 0.08;
stripe = abs(X-0.70)<0.12 & abs(Y-0.70)<0.14;
texture = 0.5 + 0.35*cos(2*pi*22*X);
original(stripe) = texture(stripe);

%% 3. 光瞳定义在空间频率域；振幅 PSF 的模方才是强度 PSF
freq = (-floor(N/2):ceil(N/2)-1)/(N*pixel_pitch_um); % cycles/um
[FX,FY] = meshgrid(freq,freq);
coherent_cutoff = 1/(2*wavelength_um*f_number);
rho2 = (FX.^2+FY.^2)/coherent_cutoff^2;
inside = rho2 <= 1;
pupil = zeros(N,N);
pupil(inside) = exp(1i*2*pi*defocus_waves*rho2(inside));
amplitude_psf = fftshift(ifft2(ifftshift(pupil)));
psf = abs(amplitude_psf).^2;
psf = psf/sum(psf,'all');
assert(all(psf>=0,'all') && abs(sum(psf,'all')-1)<1e-12);

%% 4. 非相干强度成像：y = h * x + n
% psf 的峰位在中心，先 ifftshift 再做 FFT，才能得到正确的卷积约定。
otf = fft2(ifftshift(psf));
blurred = real(ifft2(fft2(original).*otf));
rng(7);
observed = blurred + noise_std*randn(N,N); % 保存原始观测，避免截断改变噪声
restored = real(ifft2(conj(otf).*fft2(observed)./(abs(otf).^2+balance)));
observed_display = min(max(observed,0),1);
restored_display = min(max(restored,0),1);
psnr_observed = -10*log10(mean((original-observed_display).^2,'all'));
psnr_restored = -10*log10(mean((original-restored_display).^2,'all'));
assert(abs(sum(blurred,'all')-sum(original,'all'))/sum(original,'all')<1e-12);

%% 5. 六幅图使用明确的物理量和统一的强度色标
figure('Color','w','Visible','off','Position',[100 100 1200 750]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
nexttile; imagesc(original,[0 1]); axis image; colorbar;
title('Original linear intensity'); xlabel('x [pixel]'); ylabel('y [pixel]');
nexttile; phase = angle(pupil); phase(~inside)=NaN;
imagesc(freq,freq,phase,[-pi pi]); axis image; colorbar;
title('Pupil phase [rad], wrapped'); xlabel('f_x [cycles/um]'); ylabel('f_y [cycles/um]');
nexttile; imagesc(log10(max(psf/max(psf,[],'all'),1e-6)),[-6 0]);
axis image; colorbar; title('Intensity PSF: log10(h/max(h))');
xlabel('x [pixel]'); ylabel('y [pixel]');
nexttile; imagesc(blurred,[0 1]); axis image; colorbar;
title('Blurred before noise'); xlabel('x [pixel]'); ylabel('y [pixel]');
nexttile; imagesc(observed_display,[0 1]); axis image; colorbar;
title(sprintf('Observation: %.2f dB',psnr_observed)); xlabel('x [pixel]'); ylabel('y [pixel]');
nexttile; imagesc(restored_display,[0 1]); axis image; colorbar;
title(sprintf('Wiener: %.2f dB',psnr_restored)); xlabel('x [pixel]'); ylabel('y [pixel]');
colormap gray;
sgtitle(sprintf('Known-PSF restoration | w=%.2f waves | periodic boundary',defocus_waves));

%% 6. 保存数值与图片；原始浮点数组用于分析，截图仅用于展示
project_dir = fileparts(fileparts(mfilename('fullpath')));
output_dir = fullfile(project_dir,'outputs','matlab');
if ~exist(output_dir,'dir'), mkdir(output_dir); end
save(fullfile(output_dir,'lesson1_result.mat'),'original','pupil','psf','otf', ...
    'blurred','observed','restored','pixel_pitch_um','wavelength_um','f_number', ...
    'defocus_waves','noise_std','balance','psnr_observed','psnr_restored');
exportgraphics(gcf,fullfile(output_dir,'comparison.png'),'Resolution',150);
fprintf('PSF sum: %.12f\nObservation PSNR: %.3f dB\nWiener PSNR: %.3f dB\n', ...
    sum(psf,'all'),psnr_observed,psnr_restored);
fprintf('Saved: %s\n',output_dir);

% 建议每次只改一个量：w=0/0.4/0.8/1.2；noise_std=0/0.005；balance=0.001/0.01。
% w=0 依然有有限孔径衍射；图像不是绝对清晰。正负 w 的理想强度 PSF 相同。
