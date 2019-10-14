function read_file(filename, idx) {
    var reader = new FileReader();
    reader.onload = (function(i){
        console.log('processing1: ' + filename);
        return function(event){
            var b64string = event.target.result;
            console.log(event);
            file_source.data = {'file_contents' : [b64string], 'file_name': [input.files[i].name]};
            file_source.change.emit();
        };
    })(idx);
    reader.onerror = error_handler;

    // readAsDataURL represents the file's data as a base64 encoded string
    var re = /(?:\.([^.]+))?$/g;
    var ext = (re.exec(input.files[idx].name))[1];
    if (ext === "dat" || ext === "traces"){
        reader.readAsDataURL(filename);
    } else{ alert(ext + " extension found, only .dat and .traces files accepted for now")}
}

function error_handler(evt) {
    if(evt.target.error.name === "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.multiple=true;
input.onchange = function(){
    if (window.FileReader) {
        new_counter.data = {'value': [input.files.length]};
        new_counter.change.emit();
        for (var i = 0; i < input.files.length; i++){
            read_file(input.files[i], i);
        }
    } else {
        alert('FileReader is not supported in this browser');
    }
};
input.click();