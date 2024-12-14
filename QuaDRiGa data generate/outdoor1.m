% Clear workspace and set random seed
clear all;
rng(1234); 

% Turn off warnings and text output
warning('off','all');
s = qd_simulation_parameters;
s.show_progress_bars = 0;

% Simulation parameters
center_frequency = 3.5e9; % 3.5 GHz
bandwidth = 100e6; % 100 MHz
num_subcarriers = 1;
num_samples = 10000;
path_length = 100; % 100m straight path

% Create linear movement pattern
rx_x = linspace(0, path_length, num_samples)'; % Linear movement along x-axis
rx_y = 10 * ones(num_samples, 1); % Constant y position
rx_z = 1.5 * ones(num_samples, 1); % Constant height for receiver

% Combine into rx_positions
rx_positions = [rx_x'; rx_y'; rx_z'];

% Fixed TX position (base station)
tx_height = 25; % 25m high base station
tx_positions = repmat([50; 0; tx_height], 1, num_samples); % BS position

% Initialize storage
channel_freq_resp = zeros(num_subcarriers, num_samples);

% Create layout
l = qd_layout;
l.simpar = s;

% Use Outdoor LOS scenario (Urban Micro)
l.set_scenario('3GPP_38.901_UMi_LOS');
l.tx_array = qd_arrayant('3gpp-3d');  % 3GPP antenna for BS
l.rx_array = qd_arrayant('omni');     % Omnidirectional for UE

% Set parameters
l.simpar.center_frequency = center_frequency;
l.simpar.use_absolute_delays = 1;
l.simpar.show_progress_bars = 0;

fprintf('Processing: ');
progress_step = floor(num_samples/10);

% Generate channels
for i = 1:num_samples
    % Update position for this realization
    l.tx_position = tx_positions(:,i);
    l.rx_position = rx_positions(:,i);
    
    % Generate channel and get frequency response
    channel = l.get_channels();
    h = channel(1).fr(bandwidth, num_subcarriers);
    
    % Store results
    channel_freq_resp(:,i) = h(1,1,:);
    
    if mod(i, progress_step) == 0
        fprintf('.');
    end
end
fprintf(' Done!\n');

% Save binary files
fid = fopen('channel_real_out.bin', 'wb');
fwrite(fid, real(channel_freq_resp), 'double');
fclose(fid);

fid = fopen('channel_imag_out.bin', 'wb');
fwrite(fid, imag(channel_freq_resp), 'double');
fclose(fid);

% Save dimensions
dlmwrite('dimensions.txt', [size(channel_freq_resp, 1), size(channel_freq_resp, 2)], 'delimiter', '\t');

% Plot results
figure('Visible', 'on');
subplot(2,1,1);
plot(1:num_samples, abs(channel_freq_resp), 'b-', 'LineWidth', 1);
title('Channel Response Over Time (Outdoor UMi LOS)');
xlabel('Sample Index');
ylabel('Magnitude');
grid on;
xlim([1 num_samples]);

% Plot movement pattern
subplot(2,1,2);
plot(rx_x, rx_y, 'b.', 'MarkerSize', 1);
hold on;
plot(tx_positions(1,1), tx_positions(2,1), 'r*', 'MarkerSize', 10);
title('Movement Pattern (Top View)');
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;
legend('UE Path', 'BS Position', 'Location', 'northeast');
axis equal;

% Calculate and display some statistics
fprintf('\nChannel Statistics:\n');
fprintf('Mean channel magnitude: %.4f\n', mean(abs(channel_freq_resp)));
fprintf('Max channel magnitude: %.4f\n', max(abs(channel_freq_resp)));
fprintf('Min channel magnitude: %.4f\n', min(abs(channel_freq_resp)));

% Re-enable warnings
warning('on','all');