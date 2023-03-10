# Generic troubleshooting tips

- User account for license server on host1(sentinel node) is: admin Ekfz_swarm_2

- x.509 certificates are not configured correctly – See [https://www.linuxjournal.com/content/understanding-public-key-infrastructure-and-x509-certificates](https://www.linuxjournal.com/content/understanding-public-key-infrastructure-and-x509-certificates).
- License server is not running or Swarm licenses are not installed - See chapter "HPE AutoPass License Server License Management" in **AutoPass License Server User Guide** for details of the web GUI management interface and how to install license.
- Swarm core components (Docker containers) are not started or errors while starting. – For more information on how to start Swarm Learning, see [Running Swarm Learning](/docs/Install/Running_Swarm_Learning.md).
- Swarm components are not able to see each other - See the [Exposed Ports](/docs/Install/Exposed_port_numbers.md) to see if the required ports are exposed.
- User is not using the Swarm APIs correctly – See [Swarm Wheels Package](/docs/User/Swarm_client_interface-wheels_package.md) for details of API.
- Errors related to SWOP task definition, profile schema, or SWCI init script – These are user defined artifacts. Verify these files for correctness.
- Any experimental release of Ubuntu greater than LTS 20.04 may result in the following error message when running SWOP tasks.
  ```SWOP MAKE_USER_CONTAINER fails.```
  This occurs as SWOP is not able to obtain image of itself because of Docker setup differences in this experimental Ubuntu release. Switch to 20.04 LTS to resolve  this issue.

# <a name="GUID-96BB1337-2B99-45C7-BA9F-3D7D3B76663E"/> Troubleshooting

Troubleshooting provides solutions to commonly observed issues during Swarm Learning set up and execution.

## 1. <a name="GUID-EDAB2731-9CF3-4770-B54C-40C56D2FFDAC"/> Error code: 6002

```
> Error message: Unable to connect to server. Server might be wrongly configured or down.
> Custom message: Error in communicating with server https://HOST_SYSTEM_IP:5814 (default port)
```

**Problem description**

Error code: 6002, as shown in the following screenshot occurs when Swarm Learning components are not able to connect to the APLS server.![Troubleshooting_image](GUID-28273425-4E6F-425D-8A32-339013B86F75-high.png)

**Resolution**

1.  Verify if License Server is running.

    On the License Server host, verify if it is running, if not, restart the License Server.

    For more information about restarting the License Server, see *AutoPass License Server User Guide*.

2.  Access the APLS web management console. If the browser cannot connect, verify the network proxy settings, firewall policies, that are in effect. If required, work with your network administrator to resolve.

3.  Verify if the Swarm licenses are installed using APLS web management console. For more information, see APLS User Guide.


## 2. Installation of HPE Swarm Learning on air-gaped systems or if the Web UI Installer runs into any issue and not able to install

- Download the following from HPE My Support Center(MSC) on a host system that has internet access - tar file (HPE_SWARM_LEARNING_DOCS_EXAMPLES_SCRIPTS_Q2V41-11033.tar.gz) and the signature file for the above tar file.
- Untar the tar file under `/opt/hpe/swarm-learning`.
- Do a docker login from your host:
   `docker login hub.myenterpriselicense.hpe.com –u <YOUR-HPE-PASSPORT-EMAIL> -p hpe_eval`
- Pull the signed Swarm Learning images from HPEs Docker Trust Registry (DTR):
   ```
   docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sn:<latest Swarm Learning Version>
   docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sl:<latest Swarm Learning Version>
   docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/swci:<latest Swarm Learning Version>
   docker pull hub.myenterpriselicense.hpe.com/hpe/swarm-learning/swop:<latest Swarm Learning Version>
   
   For eg: docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sn:1.2.0
   ```
- Copy the tar file and Docker images to the air-gaped systems.

## 3. System resource issues if too many SLs are mapped to the same SN

When configuring Swarm Learning you may encounter system resource issues if too many SLs are mapped to same SN. For example:
    ```
    “swarm.blCnt : WARNING: SLBlackBoardObj : errCheckinNotAllowed:CHECKIN NOT ALLOWED”
    ```
The suggested workaround is to start with mapping 4 SL to 1 SN. Then after, slowly scale no of SLs to SN

## 4. SWCI waits for task-runner indefinitely even after task completed or failed

User to ensure no failure in ML code before Swarm training starts. Check using `SWARM_LOOPBACK ENV` and ensure, user coderuns fine and local training completes successfully.

## 5. Error while docker pull Swarm Learning images: 'could not rotate trust to a new trusted root'

Please remove below directories and re-try pull images: <br> </br>
~/.docker/trust/tuf/hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/swci/
~/.docker/trust/tuf/hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sn/
~/.docker/trust/tuf/hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/swop/
~/.docker/trust/tuf/hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sl/

## 6. GPU env setup
![gpu-driver-setup.png](assets%2Fgpu-driver-setup.png)

It could be very tricky to set up the gpu env for SL. First make sure

- `$ nvidia-smi` returns the driver status. `$ nvcc -V` doesn't need to be set up.
- If all the debugging fails, try to `$ sudo apt-get purge -y <all nvidia drivers/environments>`
- Reinstall nvidia driver as the picture shown
- Rerun the gpu_env_setup.sh

## 7. Docker container problem
Error message displayed when reaching the step to run sn node:

`Error response from daemon: error while creating mount source path  read-only file system`

It's because of docker-snap. Switching out to docker-ce solved this.

Q: how can I determine what Docker version I’m using and how do I uninstall Docker Snap and re-install the official version of Docker.

Here’s how:
- Run sudo snap list to see verify that Docker is installed with snap. If you see it on the list, it is.
- Then run: snap remove docker
- Reboot.
- Follow official docker install guide: https://docs.docker.com/engine/install/ubuntu/


## 8. Other hosts couldn't connect to sentinel node when running sn
Log:
```
`(base) swarm@dl1:/opt/hpe/swarm-learning$ sudo ./scripts/bin/run-sn -it --rm --name=sn2 --network=host-2-net --host-ip=192.168.33.103 --sentinel-ip=192.168.33.102 --sn-p2p-port=30303 --sn-api-port=30304 --key=workspace/mnist-pyt-gpu/cert/sn-2-key.pem --cert=workspace/mnist-pyt-gpu/cert/sn-2-cert.pem --capath=workspace/mnist-pyt-gpu/cert/ca/capath --apls-ip=192.168.33.102 --apls-port 5000
a67f842daae7e086edbf37e81722a4000336b8d11b71a5f1b62912466ddee859
######################################################################
##                    HPE SWARM LEARNING SN NODE                    ##
######################################################################
## © Copyright 2019-2022 Hewlett Packard Enterprise Development LP  ##
######################################################################
2023-02-01 10:50:16,289 : swarm.blCnt : INFO : Setting up blockchain layer for the swarm node: START
2023-02-01 10:50:17,526 : swarm.blCnt : INFO : Creating Autopass License Provider
2023-02-01 10:50:18,264 : swarm.blCnt : INFO : Creating license server
2023-02-01 10:50:18,264 : swarm.blCnt : INFO : Setting license servers
2023-02-01 10:50:18,273 : swarm.blCnt : INFO : Acquiring floating license 1100000380:1
2023-02-01 10:50:18,770 : swarm.SN : INFO : Using URL : https://192.168.33.102:30304/is_up`
```
SN node will stuck at this line for quite long and raise timeout error: couldn't connect to sentinel node.
Solustion:
- Ensure https://localhost:5000 could be accessd on sentinel node
- Log in and check the validity of licenses
- Uninstall the license server if necessary by reading the user guide

## 9. Swarm Learning container couldn't access GPU device(CUDA Runtime Error: no CUDA-capable device is detected)
Verify if the problem also appears in your local machine env or just in swarm docker env.
Try:
```
$ sudo service docker restart
$ sudo docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```
Ultimate solution: RESTART!


## 10. Stuck at running swarm nodes
- swarm network nodes
- correct logs for sentinel nodes should be like this:
  ```
  bd63c696052db1ed6edb49720fd322d4fb2d934851776948d2554229a2ff00ab
  ######################################################################
  ##                    HPE SWARM LEARNING SN NODE                    ##
  ######################################################################
  ## © Copyright 2019-2022 Hewlett Packard Enterprise Development LP  ##
  ######################################################################
  2023-02-27 15:59:04,822 : swarm.blCnt : INFO : Setting up blockchain layer for the swarm node: START
  2023-02-27 15:59:06,216 : swarm.blCnt : INFO : Creating Autopass License Provider
  2023-02-27 15:59:07,103 : swarm.blCnt : INFO : Creating license server
  2023-02-27 15:59:07,104 : swarm.blCnt : INFO : Setting license servers
  2023-02-27 15:59:07,116 : swarm.blCnt : INFO : Acquiring floating license 1100000380:1
  2023-02-27 15:59:37,235 : swarm.SN : INFO : SMLETHNode: Starting GETH ... 
  2023-02-27 15:59:47,285 : swarm.SN : WARNING : SMLETHNode: Enode list is empty: Node is standalone
  2023-02-27 16:01:57,471 : swarm.SN : INFO : SMLETHNode: Started I-am-Alive thread
  2023-02-27 16:01:57,472 : swarm.blCnt : INFO : Setting up blockchain layer for the swarm node: FINISHED
  2023-02-27 16:01:58,108 : swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
  ```

  - correct logs for swarm nodes should be like this:

  ```
  bd63c696052db1ed6edb49720fd322d4fb2d934851776948d2554229a2ff00ab
  ######################################################################
  ##                    HPE SWARM LEARNING SN NODE                    ##
  ######################################################################
  ## © Copyright 2019-2022 Hewlett Packard Enterprise Development LP  ##
  ######################################################################
  2023-02-27 15:59:04,822 : swarm.blCnt : INFO : Setting up blockchain layer for the swarm node: START
  2023-02-27 15:59:06,216 : swarm.blCnt : INFO : Creating Autopass License Provider
  2023-02-27 15:59:07,103 : swarm.blCnt : INFO : Creating license server
  2023-02-27 15:59:07,104 : swarm.blCnt : INFO : Setting license servers
  2023-02-27 15:59:07,116 : swarm.blCnt : INFO : Acquiring floating license 1100000380:1
  2023-02-27 16:01:57,471 : swarm.SN : INFO : Sentinel Node is up
  2023-02-27 16:01:57,471 : swarm.SN : INFO : SMLETHNode: Starting GETH ...
  2023-02-27 16:01:57,471 : swarm.SN : INFO : SMLETHNode: Started I-am-Alive thread
  2023-02-27 16:01:57,472 : swarm.blCnt : INFO : Setting up blockchain layer for the swarm node: FINISHED
  2023-02-27 16:01:58,108 : swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
  ```
  If the SN nodes stuck at 'swarm.SN : INFO : Sentinel Node is up', please verify the network connection between the swarm nodes and the sentinel nodes. Ensure the lab network or VPN is correctly set up.
  Else if any error message is shown about the license, please check the license server is correctly installed on Sentinel node and the license is valid and correctly shared between the Sentinel nodes and the Swarm nodes.
