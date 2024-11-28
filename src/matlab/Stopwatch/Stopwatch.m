classdef Stopwatch < matlab.mixin.Copyable
    % Stopwatch
    % 
    % A utility class for analyzing code performance through time measurement.
    % Derived from `matlab.mixin.Copyable` to allow deep copying of instances.
    %
    % Example Usage:
    %   sw = Stopwatch('My Stopwatch');
    %   sw.add('code1', 'Benchmark code snippet 1');
    %   sw.start('code1');
    %   % Execute some code here
    %   sw.stop('code1');
    %   sw.print(); % Display recorded times
    %
    % Features:
    %   - Add multiple named stopwatch slots for tracking different tasks.
    %   - Start, stop, reset, and print timing information for each slot.
    %   - Display results with formatted output.
    
    properties(Access=private)
        name = 'stopwatch';
        tics;
        time;
        caption;
        keys = {};
    end
    
    methods
        function obj=Stopwatch(name)
            p = inputParser;
            addRequired(p, 'name', @ischar);
            parse(p, name);
            
            obj.name=p.Results.name;

            obj.tics = containers.Map;
            obj.time = containers.Map;
            obj.caption = containers.Map;
        end
        
        function add(obj, id, caption)
            % adds a stopwatch slot
            
            % check types of parameter
            p = inputParser;
            addRequired(p, 'id', @ischar);
            addRequired(p, 'caption', @ischar);
            parse(p, id, caption);
            
            
            obj.caption(p.Results.id) = p.Results.caption;
            obj.time(p.Results.id) = 0;
            
            obj.keys{1, size(obj.keys,2) + 1} = p.Results.id;
        end
        
        function start(obj, id)
            % starts a specific stopwatch slot
            
            % check if id is string
            p = inputParser;
            addRequired(p, 'id', @ischar);
            parse(p, id);
            
            obj.tics(p.Results.id) = tic;
        end
        
        function stop(obj, id)
            % stops a specific stopwatch slot
            
            % check if id is string
            p = inputParser;
            addRequired(p, 'id', @ischar);
            parse(p, id);
            
            timerVal=obj.tics(p.Results.id);
            elapsedTime=toc(timerVal);
            obj.time(p.Results.id)=obj.time(p.Results.id) + elapsedTime;
        end
        
        function v=getTotal(obj)
            v=0;
            % estimate maxlen
            for id=1:size(obj.keys,2)
                key=obj.keys{id};
                v = v + obj.time(key);
            end
        end
        
        function reset(obj, id)
            % resets a specific stopwatch slot
            
            % check if id is string
            p = inputParser;
            addRequired(p, 'id', @ischar);
            parse(p, id);
            
            % reset
            obj.time(p.Results.id) = 0;
            obj.tics.remove(p.Results.id);    
        end
        
        function print(obj)
            % nicely formatted output
            
            maxLenCaption=0;
            maxLenKey=0;
            maxLenTimeSek=0;
            maxLenTimeMs=0;
            
            summedTime=0;
            
            % estimate maxlen
            for id=1:size(obj.keys,2)
                key=obj.keys{id};
                maxLenCaption = ifelse(maxLenCaption > size(obj.caption(key), 2), ...
                    maxLenCaption, size(obj.caption(key), 2) );
                maxLenKey = ifelse(maxLenKey > size(key, 2), ...
                    maxLenKey, size(key, 2) );
                
                tmp = regexp(num2str(obj.time(key)),'\.','split');
                
                strSek=cell2mat(tmp(1));
                maxLenTimeSek = ifelse(maxLenTimeSek > size(strSek, 2), ...
                maxLenTimeSek, size(strSek, 2) );


                strMs=cell2mat(tmp(2));
                maxLenTimeMs = ifelse(maxLenTimeMs > size(strMs, 2), ...
                maxLenTimeMs, size(strMs, 2) );
                
                summedTime = summedTime + obj.time(key);
            end
            
            tmp = regexp(num2str(summedTime),'\.','split');
                
            strSek=cell2mat(tmp(1));
            maxLenTimeSek = ifelse(maxLenTimeSek > size(strSek, 2), ...
                maxLenTimeSek, size(strSek, 2) );


            strMs=cell2mat(tmp(2));
            maxLenTimeMs = ifelse(maxLenTimeMs > size(strMs, 2), ...
                maxLenTimeMs, size(strMs, 2) );
            
            
            % right padding with spaces
            spacingCaptionArg = ['%-', num2str(maxLenCaption),'s'];
            % +2 because of the brackets added below (see paddedKey)
            spacingKeyArg = ['%-', num2str(maxLenKey+2),'s'];
            spacingSekArg = ['%', num2str(maxLenTimeSek),'s'];
            spacingMsArg = ['%-0', num2str(maxLenTimeMs),'s'];
            
            maxLenRow = maxLenCaption + maxLenKey + 2 + 5;
            spacingSumArg = ['%', num2str(maxLenRow), 's'];
            
            
            cprintf('Comments', [obj.name '\n\n']);
            for id=1:size(obj.keys,2)
                key=obj.keys{id};
                paddedCaption = sprintf(spacingCaptionArg, obj.caption(key));
                paddedKey = sprintf(spacingKeyArg, ['[' key ']']);
                
                tmp = regexp(num2str(obj.time(key)),'\.','split');
                strSek=cell2mat(tmp(1));
                strMs=cell2mat(tmp(2));
                
                paddedSek = sprintf(spacingSekArg, strSek);
                paddedMs = sprintf(spacingMsArg, strMs);

                cprintf('Text', [ paddedCaption '  ' paddedKey ' : ' paddedSek '.' paddedMs ' s\n']);
            end
            
            % print sum, if more than 1 timings
            if size(obj.keys,2) > 1
                
                tmp = regexp(num2str(summedTime),'\.','split');
                strSek=cell2mat(tmp(1));
                strMs=cell2mat(tmp(2));
                
                paddedSek = sprintf(spacingSekArg, strSek);
                paddedMs = sprintf(spacingMsArg, strMs);
                
                paddingLeft = sprintf(spacingSumArg, ' ');
                
                strSummedTime = [paddedSek '.' paddedMs ' s'];
                
                line=repmat('-', 1, size(num2str(strSummedTime),2));
                
                cprintf('Text', [paddingLeft line '\n']);
                cprintf('Text', [paddingLeft strSummedTime '\n']);
            end
            
        end
    end
    
end

