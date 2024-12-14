% Clear workspace and set random seed
clear all;
rng(1); %1234

% Turn off warnings and text output
warning('off','all');
s = qd_simulation_parameters;
s.show_progress_bars = 0;

% Simulation parameters
center_frequency = 3.5e9; % 3.5 GHz
bandwidth = 100e6; % 100 MHz
num_subcarriers = 1; % Single subcarrier
num_samples = 10000; % Number of channel realizations
room_size = 20; % Room size in meters

% Pre-allocate arrays for speed
tx_positions = room_size * rand(3, num_samples);
rx_positions = room_size * rand(3, num_samples);
tx_positions(3,:) = 1.5; % Set TX height to 1.5m
rx_positions(3,:) = 1.5; % Set RX height to 1.5m

% Initialize storage
channel_freq_resp = zeros(num_subcarriers, num_samples);

% Create reusable objects outside the loop
l = qd_layout;
l.simpar = s;
l.set_scenario('3GPP_38.901_Indoor_NLOS');
l.tx_array = qd_arrayant('omni');
l.rx_array = qd_arrayant('omni');
l.simpar.center_frequency = center_frequency;
l.simpar.use_absolute_delays = 1;

% Silent progress tracking
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
    
    % Show minimal progress indicator
    if mod(i, progress_step) == 0
        fprintf('.');
    end
end
fprintf(' Done!\n');

% Save binary files without displaying messages
fid = fopen('channel_real_1.bin', 'wb');
fwrite(fid, real(channel_freq_resp), 'double');
fclose(fid);

fid = fopen('channel_imag_1.bin', 'wb');
fwrite(fid, imag(channel_freq_resp), 'double');
fclose(fid);

% Save dimensions silently
dlmwrite('dimensions.txt', [size(channel_freq_resp, 1), size(channel_freq_resp, 2)], 'delimiter', '\t');

% Plot results without displaying processing info
figure('Visible', 'on');
plot(1:num_samples, abs(channel_freq_resp), 'b-', 'LineWidth', 1);
title('Channel Response Over Time');
xlabel('Sample Index');
ylabel('Magnitude');
grid on;
xlim([1 num_samples]);

% Re-enable warnings
warning('on','all');