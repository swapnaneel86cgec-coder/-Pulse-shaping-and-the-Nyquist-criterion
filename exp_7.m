% Experiment 7 - Pulse Shaping and the Nyquist Criterion
% Implementation of RC/RRC filters, visualizations, and validation.

clear; clc; close all;

%% 1. Simulation Parameters
T  = 1;                    % Symbol duration
N  = 10;                   % Oversampling factor (samples per symbol)
Fs = N / T;                % Sampling frequency
t  = -5*T : 1/Fs : 5*T;    % Time vector (-5T to +5T)
alphas = [0, 0.25, 0.5, 1];% Roll-off factors to test (Task 24)

f = linspace(-Fs/2, Fs/2, length(t)); % Frequency axis for FFT plots

%% 2. Generate and Plot Pulse & Frequency Responses (Tasks 24 & 25)
figure('Name', 'Filter Responses', 'Position', [100, 100, 900, 400]);

for i = 1:length(alphas)
    alpha = alphas(i);

    % Generate mathematically defined RC filter
    h_rc = generate_RC(t, alpha, T);

    % Time Domain Plot
    subplot(1, 2, 1);
    plot(t, h_rc, 'LineWidth', 1.5, 'DisplayName', ['\alpha = ', num2str(alpha)]);
    hold on;

    % Frequency Domain Plot (Magnitude)
    H_rc = fftshift(abs(fft(h_rc))) / Fs;
    subplot(1, 2, 2);
    plot(f, H_rc, 'LineWidth', 1.5, 'DisplayName', ['\alpha = ', num2str(alpha)]);
    hold on;
end

% Formatting Time Domain
subplot(1, 2, 1);
title('Time Domain: Raised Cosine Pulse');
xlabel('Time (t/T)'); ylabel('Amplitude');
grid on; legend; xlim([-5 5]);

% Formatting Frequency Domain
subplot(1, 2, 2);
title('Frequency Domain: Bandwidth vs. Roll-off');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
grid on; legend; xlim([-1 1]);

%% 3. Bandwidth vs. Roll-off Factor (Required Visualization)
alphas_sweep   = 0:0.05:1;
bw_analytical  = (1 + alphas_sweep) / (2*T);
bw_measured    = nan(size(alphas_sweep));

threshold = 0.01;                 % 1% of peak magnitude
pos_idx   = f >= 0;               % one-sided (positive) frequency axis
f_pos     = f(pos_idx);

for i = 1:length(alphas_sweep)
    h_temp   = generate_RC(t, alphas_sweep(i), T);
    H_temp   = fftshift(abs(fft(h_temp)));
    H_norm   = H_temp / max(H_temp);      % normalize to peak = 1
    H_pos    = H_norm(pos_idx);

    edge_idx = find(H_pos < threshold, 1, 'first');
    if ~isempty(edge_idx)
        bw_measured(i) = f_pos(edge_idx);
    end
end

figure('Name', 'Bandwidth vs Roll-off', 'Position', [120, 120, 600, 400]);
plot(alphas_sweep, bw_analytical, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Analytical: (1+\alpha)/(2T)');
hold on;
plot(alphas_sweep, bw_measured, 'r--x', 'LineWidth', 1.5, 'DisplayName', 'Measured (FFT, 1% threshold)');
title('Occupied Bandwidth vs. Roll-off Factor \alpha');
xlabel('Roll-off factor \alpha'); ylabel('Bandwidth (Hz)');
grid on; legend('Location', 'northwest');

%% 4. Cascade Transmitter and Receiver RRC Filters (Task 27)
alpha_test = 0.5; % Using alpha = 0.5 for the cascade demonstration
h_rrc_tx = generate_RRC(t, alpha_test, T);
h_rrc_rx = generate_RRC(t, alpha_test, T);

% Convolve Tx and Rx filters (Dividing by Fs normalizes the discrete convolution)
h_cascaded  = conv(h_rrc_tx, h_rrc_rx, 'same') / Fs;
h_rc_ideal  = generate_RC(t, alpha_test, T); % Ideal RC for comparison

figure('Name', 'Cascaded RRC vs RC', 'Position', [150, 150, 600, 400]);
plot(t, h_rc_ideal, 'b', 'LineWidth', 2, 'DisplayName', 'Theoretical RC'); hold on;
plot(t, h_cascaded, 'r--', 'LineWidth', 2, 'DisplayName', 'Cascaded RRC Tx+Rx');
title(['Cascaded RRC Filters vs Theoretical RC (\alpha = ', num2str(alpha_test), ')']);
xlabel('Time (t/T)'); ylabel('Amplitude');
grid on; legend;

%% 5. Upsampled Symbol Train and Pulse Shaping (Task 23 & Plotting)
num_symbols      = 20;
symbols          = randsrc(1, num_symbols, [-1, 1]);  % Random +1/-1 symbols
upsampled_train  = upsample(symbols, N);                % Insert N-1 zeros between symbols
t_stream         = (0:length(upsampled_train)-1) / Fs; % Time vector for stream

% Convolve stream with cascaded filter
tx_signal = conv(upsampled_train, h_cascaded, 'same');

figure('Name', 'Pulse-Shaped Stream', 'Position', [200, 200, 800, 300]);
stem(t_stream(1:N:end), symbols, 'k', 'LineWidth', 1.5, 'DisplayName', 'Original Symbols'); hold on;
plot(t_stream, tx_signal, 'r', 'LineWidth', 1.5, 'DisplayName', 'Pulse-Shaped Waveform');
title('Pulse-Shaped Data Stream');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; legend;

%% 6. Mandatory Validation: Tabulate Nyquist Zero Crossings (Task 26)
fprintf('\n--- MANDATORY VALIDATION: Nyquist Zero Crossings ---\n');
fprintf('Alpha = %g\n', alpha_test);
fprintf('----------------------------------------------------\n');
fprintf('Time (t)\t|\tAmplitude (Ideal 1 at t=0, 0 elsewhere)\n');
fprintf('----------------------------------------------------\n');

[~, center_idx] = min(abs(t));

for k = -2:2
    idx       = center_idx + (k * N);
    time_val  = t(idx);
    amplitude = h_rc_ideal(idx);
    fprintf('%dT \t\t|\t%.6f\n', k, amplitude);
end
fprintf('----------------------------------------------------\n\n');

%% --- LOCAL HELPER FUNCTIONS ---

function h = generate_RC(t, alpha, T)
    if nargin < 3
        error('generate_RC needs 3 inputs (t, alpha, T). Run the main script instead.');
    end

    h = zeros(size(t));

    if alpha == 0
        h = sinc(t / T);
        return;
    end

    is_singular = abs(abs(t) - T/(2*alpha)) < 1e-6;
    is_normal   = ~is_singular;

    h(is_singular) = (pi/4) * sinc(1/(2*alpha));

    num = cos(pi * alpha * t(is_normal) / T);
    den = 1 - (2 * alpha * t(is_normal) / T).^2;
    h(is_normal) = sinc(t(is_normal) / T) .* (num ./ den);
end

function h = generate_RRC(t, alpha, T)
    if nargin < 3
        error('generate_RRC needs 3 inputs (t, alpha, T). Run the main script instead.');
    end

    h = zeros(size(t));
    is_zero = abs(t) < 1e-9;

    if alpha ~= 0
        is_edge = abs(abs(t) - T/(4*alpha)) < 1e-6;
    else
        is_edge = false(size(t));
    end

    is_normal = ~is_zero & ~is_edge;

    h(is_zero) = 1 - alpha + (4*alpha)/pi;

    if any(is_edge)
        h(is_edge) = (alpha/sqrt(2)) * ((1+2/pi)*sin(pi/(4*alpha)) + (1-2/pi)*cos(pi/(4*alpha)));
    end

    tn  = t(is_normal);
    num = sin(pi * tn * (1-alpha)/T) + 4 * alpha * (tn/T) .* cos(pi * tn * (1+alpha)/T);
    den = (pi * tn / T) .* (1 - (4 * alpha * tn / T).^2);
    h(is_normal) = num ./ den;
end