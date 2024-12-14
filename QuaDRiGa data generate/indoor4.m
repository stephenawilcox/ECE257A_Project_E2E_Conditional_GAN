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
room_size = 20; % 10m room size

% Create a more realistic walking pattern (random walk with preferred direction)
dx = 0.2 * randn(num_samples, 1); % Random steps in x
dy = 0.2 * randn(num_samples, 1); % Random steps in y

% Initialize positions
rx_x = zeros(num_samples, 1);
rx_y = zeros(num_samples, 1);
rx_z = 1.5 * ones(num_samples, 1);

% Starting position
rx_x(1) = 2;
rx_y(1) = 2;

% Generate random walk with boundaries and obstacle avoidance
for i = 2:num_samples
    % Calculate next position
    next_x = rx_x(i-1) + dx(i);
    next_y = rx_y(i-1) + dy(i);
    
    % Keep within room boundaries
    next_x = max(0.5, min(room_size-0.5, next_x));
    next_y = max(0.5, min(room_size-0.5, next_y));
    
    % Store positions
    rx_x(i) = next_x;
    rx_y(i) = next_y;
end

% Combine into rx_positions
rx_positions = [rx_x'; rx_y'; rx_z'];

% Fixed TX position (ceiling mounted AP)
tx_positions = repmat([room_size/2; room_size/2; 2.5], 1, num_samples);

% Initialize storage
channel_freq_resp = zeros(num_subcarriers, num_samples);

% Create layout
l = qd_layout;
l.simpar = s;

% Use Indoor NLOS scenario (more realistic with obstacles)
l.set_scenario('3GPP_38.901_Indoor_NLOS');
l.tx_array = qd_arrayant('omni');
l.rx_array = qd_arrayant('omni');

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
fid = fopen('channel_real_3.bin', 'wb');
fwrite(fid, real(channel_freq_resp), 'double');
fclose(fid);

fid = fopen('channel_imag_3.bin', 'wb');
fwrite(fid, imag(channel_freq_resp), 'double');
fclose(fid);

% Save dimensions
dlmwrite('dimensions.txt', [size(channel_freq_resp, 1), size(channel_freq_resp, 2)], 'delimiter', '\t');

% Plot results
figure('Visible', 'on');
subplot(2,1,1);
plot(1:num_samples, abs(channel_freq_resp), 'b-', 'LineWidth', 1);
title('Channel Response Over Time (Indoor NLOS with Obstacles)');
xlabel('Sample Index');
ylabel('Magnitude');
grid on;
xlim([1 num_samples]);

% Plot movement pattern with obstacles
subplot(2,1,2);
plot(rx_x, rx_y, 'b.', 'MarkerSize', 1);
hold on;
plot(tx_positions(1,1), tx_positions(2,1), 'r*', 'MarkerSize', 10);

% Add virtual obstacles (for visualization)
rectangle('Position',[3 3 1 1],'FaceColor',[0.7 0.7 0.7]);
rectangle('Position',[6 6 1 1],'FaceColor',[0.7 0.7 0.7]);
rectangle('Position',[7 3 1 1],'FaceColor',[0.7 0.7 0.7]);

title('RX Movement Pattern with Obstacles (Top View)');
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;
legend('RX Path', 'TX Position', 'Location', 'northeast');
axis equal;
xlim([0 room_size]);
ylim([0 room_size]);

% Re-enable warnings
warning('on','all');