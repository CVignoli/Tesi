function [tgtLm, type] = readGTlandmarks(filename)
fid = fopen(filename);
    % Skip two header lines
    tline = fgetl(fid);
    tline = fgetl(fid);
    tline = fgetl(fid);
    tgtLm = [];
    type = {};
    while ischar(tline)
        newstr = split(tline);
        coord = split(newstr{3},',');
        coord = cell2mat(cellfun(@str2num,coord,'UniformOutput',false));
        tline = fgetl(fid);
        tgtLm = [tgtLm; coord'];
        type = [type; newstr{end}];
    end
    fclose(fid);
end

