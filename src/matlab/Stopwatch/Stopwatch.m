classdef Stopwatch < matlab.mixin.Copyable
    % STOPWATCH Class to benchmark and analyze code performance.
    %   This class is useful for timing and logging code execution in MATLAB.
    %   By setting up named time slots, you can easily measure the duration
    %   of different parts of your code and print formatted results.
    %
    %   Usage:
    %     sw = Stopwatch('My Stopwatch');
    %     sw.add('task1', 'Description of Task 1');
    %     sw.start('task1');
    %     % Place code for Task 1 here
    %     sw.stop('task1');
    %     sw.print();
    %
    % Properties (Private):
    %   name       - Name of the stopwatch instance.
    %   tics       - Map container storing start times for each task.
    %   time       - Map container storing accumulated times for each task.
    %   caption    - Map container storing task descriptions.
    %   keys       - Cell array storing task identifiers.
    %
    % Methods:
    %   Stopwatch         - Constructor to initialize a new stopwatch with a specified name.
    %   add               - Adds a new timing slot with an ID and description.
    %   start             - Starts timing for the specified slot.
    %   stop              - Stops timing for the specified slot and accumulates elapsed time.
    %   getTotal          - Returns the total accumulated time across all slots.
    %   reset             - Resets the accumulated time for a specific slot.
    %   print             - Prints a formatted report of all timing slots.
    %
    % Usage Notes:
    %   - Each timing slot has a unique ID used to start, stop, and reset timings.
    %   - The `print` method outputs results in a readable format, including the total time if more than one slot is present.
    %
    % Example:
    %   sw = Stopwatch('Benchmark');
    %   sw.add('loop', 'Loop Execution');
    %   sw.start('loop');
    %   for i = 1:1000
    %       % Code to be timed
    %   end
    %   sw.stop('loop');
    %   sw.print();
    
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

