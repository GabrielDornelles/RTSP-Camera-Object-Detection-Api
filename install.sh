green=`tput setaf 2`
reset=`tput sgr0`
pv=`python3 -V`
echo "${green}==== Creating Venv ====${reset}"
echo "Using ${pv}"
python3 -m venv venv
PWD=`pwd`
activate () {
    . $PWD/venv/bin/activate
}
activate
echo "${green}==== Installing packages ====${reset}"
pip3 install -r requirements.txt
echo "run: >>> uvicorn app:app"
echo "run with WSGI (example): >>> gunicorn -b 127.0.0.1:5000 --daemon -k uvicorn.workers.UvicornH11Worker app:app"
