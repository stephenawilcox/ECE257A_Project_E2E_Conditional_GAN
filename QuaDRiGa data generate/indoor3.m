% Clear workspace and set random seed
clear all;
rng(1234); 

% Turn off warnings and text output
warning('off','all');
s = qd_simulation_parameters;
s.show_progress_bars = 0;

% Simulation parameters for realistic indoor
center_frequency = 3.5e9; % 3.5 GHz
bandwidth = 100e6; % 100 MHz
num_subcarriers = 1;
num_samples = 10000;
room_size = 20; % Smaller room size (10m) for more realistic indoor

% Create more realistic movement pattern (walking path)
t = linspace(0, 2*pi, num_samples)';
radius = 2; % 2m radius walking pattern
rx_x = room_size/2 + radius * cos(t);
rx_y = room_size/2 + radius * sin(t);
rx_z = 1.5 * ones(num_samples, 1); % Constant height

% Combine into rx_positions
rx_positions = [rx_x'; rx_y'; rx_z'];

% Fixed TX position (like a WiFi access point)
tx_positions = repmat([room_size/2; room_size/2; 2.5], 1, num_samples); % AP at 2.5m height

% Initialize storage
channel_freq_resp = zeros(num_subcarriers, num_samples);

% Create reusable objects outside the loop
l = qd_layout;
l.simpar = s;

% Use standard Indoor LOS scenario
l.set_scenario('3GPP_38.901_Indoor_LOS'); % Using supported scenario
l.tx_array = qd_arrayant('omni');
l.rx_array = qd_arrayant('omni');

% Set specific parameters
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
fid = fopen('channel_real_2_test.bin', 'wb');
fwrite(fid, real(channel_freq_resp), 'double');
fclose(fid);

fid = fopen('channel_imag_2_test.bin', 'wb');
fwrite(fid, imag(channel_freq_resp), 'double');
fclose(fid);

% Save dimensions
dlmwrite('dimensions.txt', [size(channel_freq_resp, 1), size(channel_freq_resp, 2)], 'delimiter', '\t');

% Plot results
figure('Visible', 'on');
subplot(2,1,1);
plot(1:num_samples, abs(channel_freq_resp), 'b-', 'LineWidth', 1);
title('Channel Response Over Time (Indoor LOS)');
xlabel('Sample Index');
ylabel('Magnitude');
grid on;
xlim([1 num_samples]);

% Add movement visualization
subplot(2,1,2);
plot(rx_x, rx_y, 'b.', 'MarkerSize', 1);
hold on;
plot(tx_positions(1,1), tx_positions(2,1), 'r*', 'MarkerSize', 10);
title('RX Movement Pattern (Top View)');
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;
legend('RX Path', 'TX Position');
axis equal;

% Re-enable warnings
warning('on','all');