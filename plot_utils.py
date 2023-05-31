import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

import numpy as np
from scipy.stats import gaussian_kde

plt.set_loglevel("critical")

def plot_particles(particles1, particles2=None, ln_prob=None, plot_particles=True,
                   plot_particles_density=False, same_axes=True, mesh_size=100,
                   title=None, save=False, display=True, filename=None, plot_dir=None,
                   label1="SVGD", label2="Coin SVGD", dims=2, one_dim_particles="flat",
                   **plt_kwargs):

    if dims == 1:
        if particles2 is None or same_axes is True:
            fig, ax = new_plot(compare=False, dims=1, **plt_kwargs)
            if ln_prob is not None:
                ax = plot_target(ln_prob, ax, mesh_size, dims=1)
            xx, dx = plot_grid(ax, mesh_size, dims=1)
            particles1_density = gaussian_kde(particles1[:, 0], bw_method=0.2)
            particles2_density = gaussian_kde(particles2[:, 0], bw_method=0.2)
            if one_dim_particles == "target" and ln_prob is not None:
                normalising_constant = np.sum(np.exp(ln_prob(xx)) * dx)
            if plot_particles:
                if one_dim_particles == "flat":
                    ax.plot(particles1[:, 0], [0]*particles1.shape[0], ".", markersize=10, color="red", label=label1)
                elif one_dim_particles == "target" and ln_prob is not None:
                    ax.plot(particles1[:, 0], np.exp(ln_prob(particles1[:, 0])) / normalising_constant, ".",
                            markersize=10, color="red", label=label1)
                else:
                    ax.plot(particles1[:, 0], particles1_density(particles1[:, 0]), ".", markersize=10, color="red",
                            label=label1)
                if particles2 is not None:
                    if one_dim_particles == "flat":
                        ax.plot(particles2[:, 0], [0]*particles2.shape[0], ".", markersize=10, color="green", label=label1)
                    elif one_dim_particles == "target" and ln_prob is not None:
                        ax.plot(particles2[:, 0], np.exp(ln_prob(particles2[:, 0])) / normalising_constant, ".",
                                markersize=10, color="green ", label=label1)
                    else:
                        ax.plot(particles2[:, 0], particles2_density(particles2[:, 0]), ".", markersize=10,
                                color="green", label=label2)
            if plot_particles_density:
                ax.plot(xx, particles1_density(xx), color="red")
                if particles2 is not None:
                    ax.plot(xx, particles2_density(xx), color="green")
            red_patch = mpatches.Patch(color='red', label=label1)
            green_patch = mpatches.Patch(color='green', label=label2)
            plt.grid(color="whitesmoke")
            plt.legend(handles=[red_patch, green_patch], prop={'size':20})

        else:
            fig, axs = new_plot(compare=True, dims=1, **plt_kwargs)
            if ln_prob is not None:
                axs[0] = plot_target(ln_prob, axs[0], mesh_size, dims=1)
                axs[1] = plot_target(ln_prob, axs[1], mesh_size, dims=1)
            xx, dx = plot_grid(axs[0], mesh_size, dims=1)
            if plot_particles_density:
                particles1_density = gaussian_kde(particles1[:, 0], bw_method=0.2)
                particles2_density = gaussian_kde(particles2[:, 0], bw_method=0.2)
            if one_dim_particles == "target" and ln_prob is not None:
                normalising_constant = np.sum(np.exp(ln_prob(xx)) * dx)
            if plot_particles:
                if one_dim_particles == "flat":
                    axs[0].plot(particles1[:, 0], [0]*particles1.shape[0], ".", markersize=10, color="red", label=label1)
                    axs[1].plot(particles2[:, 0], [0]*particles2.shape[0], ".", markersize=10, color="green", label=label1)
                elif one_dim_particles == "target" and ln_prob is not None:
                    axs[0].plot(particles1[:, 0], np.exp(ln_prob(particles1[:, 0])) / normalising_constant, ".",
                                markersize=10, color="red", label=label1)
                    axs[1].plot(particles2[:, 0], np.exp(ln_prob(particles2[:, 0])) / normalising_constant, ".",
                                markersize=10, color="green", label=label1)
                else:
                    axs[0].plot(particles1[:, 0], particles1_density(particles1[:, 0]), ".", markersize=10,
                                color="red", label=label1)

                    axs[1].plot(particles2[:, 0], particles2_density(particles2[:, 0]), ".", markersize=10,
                                color="green", label=label2)
            axs[0].grid(color='whitesmoke')
            axs[1].grid(color='whitesmoke')

            if plot_particles_density:
                axs[0].plot(xx, particles1_density(xx), color="red")
                axs[1].plot(xx, particles2_density(xx), color="green")
            red_patch = mpatches.Patch(color='red', label=label1)
            axs[0].legend(handles=[red_patch])
            green_patch = mpatches.Patch(color='green', label=label2)
            axs[1].legend(handles=[green_patch])

    if dims == 2:
        if particles2 is None or same_axes is True:
            fig, ax = new_plot(compare=False, **plt_kwargs)
            if ln_prob is not None:
                ax = plot_target(ln_prob, ax, mesh_size)
            if plot_particles:
                ax.plot(particles1[:, 0], particles1[:, 1], 'o', color="red", label=label1)
                if particles2 is not None:
                    ax.plot(particles2[:, 0], particles2[:, 1], 'o', color="green", label=label2)
            if plot_particles_density:
                mesh_x, mesh_y, xx, yy, coords = plot_grid(ax, mesh_size)
                particles1_density = gaussian_kde(particles1.T)
                z1 = particles1_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                ax.contour(mesh_x, mesh_y, z1, levels=15, colors="red")
                if particles2 is not None:
                    particles2_density = gaussian_kde(particles2.T)
                    z2 = particles2_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                    ax.contour(mesh_x, mesh_y, z2, levels=15, colors="green")
            red_patch = mpatches.Patch(color='red', label=label1)
            green_patch = mpatches.Patch(color='green', label=label2)
            plt.legend(handles=[red_patch, green_patch], prop={'size':15})

        else:
            fig, axs = new_plot(compare=True, **plt_kwargs)
            if ln_prob is not None:
                axs[0] = plot_target(ln_prob, axs[0], mesh_size)
                axs[1] = plot_target(ln_prob, axs[1], mesh_size)
            if plot_particles:
                axs[0].plot(particles1[:, 0], particles1[:, 1], 'o', color="red",label=label1)
                axs[1].plot(particles2[:, 0], particles2[:, 1], 'o', color="green",label=label2)
            if plot_particles_density:
                mesh_x, mesh_y, xx, yy, coords = plot_grid(axs[0], mesh_size)
                particles1_density = gaussian_kde(particles1.T)
                z1 = particles1_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                axs[0].contour(mesh_x, mesh_y, z1, levels=15, colors="red")
                if particles2 is not None:
                    particles2_density = gaussian_kde(particles2.T)
                    z2 = particles2_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                    axs[1].contour(mesh_x, mesh_y, z2, levels=15, colors="green")
                mesh_x, mesh_y, xx, yy, coords = plot_grid(axs[0], mesh_size)
                particles1_density = gaussian_kde(particles1.T)
                z1 = particles1_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                axs[0].contour(mesh_x, mesh_y, z1, levels=10, colors="red")
                particles2_density = gaussian_kde(particles2.T)
                z2 = particles2_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                axs[1].contour(mesh_x, mesh_y, z2, levels=10, colors="green")
            red_patch = mpatches.Patch(color='red', label=label1)
            axs[0].legend(handles=[red_patch], prop={'size':15})
            green_patch = mpatches.Patch(color='green', label=label2)
            axs[1].legend(handles=[green_patch], prop={'size':15})

    if save:
        save_plot(filename, plot_dir)

    if display:
        plt.show()


def animate_particles(all_particles1, all_particles2=None, ln_prob=None, plot_particles=True,
                      plot_particles_density=False, same_axes=True, mesh_size=100,
                      title=None, frames=None, save=False, display=True, filename=None,
                      ani_dir=None, label1="SVGD", label2="Coin-SVGD", dims=2, one_dim_particles="flat",
                      **plt_kwargs):
    if frames is None:
        frames = range(0, len(all_particles1), 50)

    if dims == 1:
        if all_particles2 is None or same_axes is True:
            fig, ax = plt.subplots()

            def animate(i, label1=label1, label2=label2):
                fig.clear()
                ax = fig.add_subplot(111, autoscale_on=False, **plt_kwargs)
                if ln_prob is not None:
                    ax = plot_target(ln_prob, ax, mesh_size, dims=1)
                xx, dx = plot_grid(ax, mesh_size, dims=1)
                particles1_density = gaussian_kde(all_particles1[i][:, 0], bw_method=0.2)
                particles2_density = gaussian_kde(all_particles2[i][:, 0], bw_method=0.2)
                if one_dim_particles == "target" and ln_prob is not None:
                    normalising_constant = np.sum(np.exp(ln_prob(xx) * dx))
                if plot_particles:
                    if one_dim_particles == "flat":
                        s = ax.plot(all_particles1[i][:, 0], [0]*all_particles1[i].shape[0], ".", markersize=10,
                                    color="red", label=label1)
                    elif one_dim_particles == "target" and ln_prob is not None:
                        s = ax.plot(all_particles1[i][:, 0], np.exp(ln_prob(all_particles1[i][:, 0])) / normalising_constant,
                                    ".", markersize=10, color="red", label=label1)
                    else:
                        s = ax.plot(all_particles1[i][:, 0], particles1_density(all_particles1[i][:, 0]), ".",
                                    markersize=10, color="red", label=label1)
                    if all_particles2 is not None:
                        if one_dim_particles == "flat":
                            s = ax.plot(all_particles2[i][:, 0], [0] * all_particles2[i].shape[0], ".", markersize=10,
                                        color="green", label=label1)
                        elif one_dim_particles == "target" and ln_prob is not None:
                            s = ax.plot(all_particles2[i][:, 0], np.exp(ln_prob(all_particles2[i][:, 0])) / normalising_constant,
                                        ".", markersize=10, color="green", label=label1)
                        else:
                            s = ax.plot(all_particles2[i][:, 0], particles2_density(all_particles2[i][:, 0]), ".",
                                        markersize=10, color="green", label=label2)
                if plot_particles_density:
                    s = ax.plot(xx, particles1_density(xx), color="red")
                    if all_particles2 is not None:
                        s = ax.plot(xx, particles2_density(xx), color="green")
                red_patch = mpatches.Patch(color='red', label=label1)
                green_patch = mpatches.Patch(color='green', label=label2)
                plt.legend(handles=[red_patch, green_patch], prop={'size':20})
                title = "Iteration: " + str(i)
                plt.suptitle(title)
                return s

        else:
            fig, axs = plt.subplots(1, 2)

            def animate(i, label1=label1, label2=label2):
                fig.clear()
                axs[0] = fig.add_subplot(121, autoscale_on=False, **plt_kwargs)
                axs[1] = fig.add_subplot(122, autoscale_on=False, **plt_kwargs)
                if ln_prob is not None:
                    axs[0] = plot_target(ln_prob, axs[0], mesh_size, dims=1)
                    axs[1] = plot_target(ln_prob, axs[1], mesh_size, dims=1)
                xx, dx = plot_grid(axs[0], mesh_size, dims=1)
                if plot_particles_density:
                    particles1_density = gaussian_kde(all_particles1[i][:, 0], bw_method=0.2)
                    particles2_density = gaussian_kde(all_particles2[i][:, 0], bw_method=0.2)
                if one_dim_particles == "target" and ln_prob is not None:
                    normalising_constant = np.sum(np.exp(ln_prob(xx)) * dx)
                if plot_particles:
                    if one_dim_particles == "flat":
                        s = axs[0].plot(all_particles1[i][:, 0],[0]*all_particles1[i].shape[0], ".",
                                        markersize=10, color="red", label=label1)
                        s = axs[1].plot(all_particles2[i][:, 0], [0]*all_particles2[i].shape[0], ".",
                                        markersize=10, color="green", label=label1)

                    elif one_dim_particles == "target" and ln_prob is not None:
                        s = axs[0].plot(all_particles1[i][:, 0], np.exp(ln_prob(all_particles1[i][:, 0])) / normalising_constant, ".",
                                    markersize=10, color="red", label=label1)
                        s = axs[1].plot(all_particles2[i][:, 0], np.exp(ln_prob(all_particles2[i][:, 0])) / normalising_constant, ".",
                                    markersize=10, color="green", label=label1)
                    else:
                        s = axs[0].plot(all_particles1[i][:, 0], particles1_density(all_particles1[i][:, 0]), ".",
                                    markersize=10, color="red", label=label1)
                        s = axs[1].plot(all_particles2[i][:, 0], particles2_density(all_particles2[i][:, 0]), ".",
                                    markersize=10, color="green", label=label2)
                if plot_particles_density:
                    s = axs[0].plot(xx, particles1_density(xx), color="red")
                    s = axs[1].plot(xx, particles2_density(xx), color="green")
                red_patch = mpatches.Patch(color='red', label=label1)
                axs[0].legend(handles=[red_patch], prop={'size':15})
                green_patch = mpatches.Patch(color='green', label=label2)
                axs[1].legend(handles=[green_patch], prop={'size':15})
                title = "Iteration: " + str(i)
                plt.suptitle(title)
                return s

    if dims == 2:
        if all_particles2 is None or same_axes is True:
            fig, ax = plt.subplots()

            def animate(i,label1=label1,label2=label2):
                fig.clear()
                ax = fig.add_subplot(111, autoscale_on=False, aspect='equal', **plt_kwargs)
                if ln_prob is not None:
                    ax = plot_target(ln_prob, ax, mesh_size)
                if plot_particles:
                    s = ax.plot(all_particles1[i][:, 0], all_particles1[i][:, 1], 'o', color="red")
                    if all_particles2 is not None:
                        s = ax.plot(all_particles2[i][:, 0], all_particles2[i][:, 1], 'o', color="green")
                if plot_particles_density:
                    mesh_x, mesh_y, xx, yy, coords = plot_grid(ax, mesh_size)
                    particles1_density = gaussian_kde(all_particles1[i].T)
                    z1 = particles1_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                    s = ax.contour(mesh_x, mesh_y, z1, levels=10, colors="red")
                    if all_particles2 is not None:
                        particles2_density = gaussian_kde(all_particles2[i].T)
                        z2 = particles2_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                        s = ax.contour(mesh_x, mesh_y, z2, levels=10, colors="green")
                red_patch = mpatches.Patch(color='red', label=label1)
                green_patch = mpatches.Patch(color='green', label=label2)
                plt.legend(handles=[red_patch, green_patch], prop={'size':20})
                title = "Iteration: " + str(i)
                plt.suptitle(title)
                return s

        else:
            fig, axs = plt.subplots(1, 2)

            def animate(i,label1=label1,label2=label2):
                fig.clear()
                axs[0] = fig.add_subplot(121, autoscale_on=False, aspect='equal', **plt_kwargs)
                axs[1] = fig.add_subplot(122, autoscale_on=False, aspect='equal', **plt_kwargs)
                if ln_prob is not None:
                    axs[0] = plot_target(ln_prob, axs[0], mesh_size)
                    axs[1] = plot_target(ln_prob, axs[1], mesh_size)
                if plot_particles:
                    s = axs[0].plot(all_particles1[i][:, 0], all_particles1[i][:, 1], 'o', color="red")
                    s = axs[1].plot(all_particles2[i][:, 0], all_particles2[i][:, 1], 'o', color="green")
                if plot_particles_density:
                    mesh_x, mesh_y, xx, yy, coords = plot_grid(axs[0], mesh_size)
                    particles1_density = gaussian_kde(all_particles1[i].T)
                    z1 = particles1_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                    s = axs[0].contour(mesh_x, mesh_y, z1, levels=10, colors="red")
                    particles2_density = gaussian_kde(all_particles2[i].T)
                    z2 = particles2_density(np.c_[xx.flatten(), yy.flatten()].T).reshape(xx.shape)
                    s = axs[1].contour(mesh_x, mesh_y, z2, levels=10, colors="green")
                red_patch = mpatches.Patch(color='red', label=label1)
                axs[0].legend(handles=[red_patch], prop={'size':15})
                green_patch = mpatches.Patch(color='green', label=label2)
                axs[1].legend(handles=[green_patch], prop={'size':15})
                title = "Iteration: " + str(i)
                plt.suptitle(title)
                return s

    ani = animation.FuncAnimation(fig, animate, fargs=(label1, label2, ), frames=frames)
    #fig.tight_layout()

    if save:
        save_ani(ani, filename, ani_dir)
    if display:
        plt.show()


def animate_multi_particles_2d(all_particles_list, ln_prob=None, plot_particles=True, mesh_size=100,
                               frames=None, save=False, display=True, filename=None, ani_dir=None,
                               label1="SVGD", label2="Coin-SVGD", x_labels=None, **plt_kwargs):
    if frames is None:
        frames = range(0, len(all_particles_list[0]), 50)

    if x_labels is None:
        x_labels = [""] * len(all_particles_list)

    fig, axs = plt.subplots(1, len(all_particles_list), figsize = (4*len(all_particles_list), 4))

    def animate(i, label1=label1, label2=label2):
        fig.clear()
        for j in range(len(all_particles_list)):
            pos = str(1) + str(len(all_particles_list)) + str(j+1)
            pos = int(pos)
            axs[j] = fig.add_subplot(pos, autoscale_on=False, aspect='equal',**plt_kwargs)
            if ln_prob is not None:
                axs[j] = plot_target(ln_prob, axs[j], mesh_size)
            if plot_particles:
                if j == 0:
                    col = "green"
                    label = "Coin SVGD"
                else:
                    col = "red"
                    label = "SVGD"
                s = axs[j].plot(all_particles_list[j][i][:, 0], all_particles_list[j][i][:, 1], 'o', color=col)
                patch = mpatches.Patch(color=col, label=label)
            axs[j].legend(handles=[patch], prop={'size':15}, loc='upper right')
            axs[j].set_xlabel(x_labels[j])
            title = "Iteration: " + str(i)
            plt.suptitle(title)
        return s

    ani = animation.FuncAnimation(fig, animate, fargs=(label1, label2, ), frames=frames)

    if save:
        save_ani(ani, filename, ani_dir)
    if display:
        plt.show()


def plot_target(ln_prob, ax, mesh_size=100, dims=2):
    if dims == 1:
        xx, dx = plot_grid(ax, mesh_size, dims=1)
        logy = ln_prob(xx)
        y = np.exp(logy)/np.sum(np.exp(logy)*dx)
        ax.plot(xx, y)

    if dims == 2:
        mesh_x, mesh_y, xx, yy, coords = plot_grid(ax, mesh_size)
        logz = ln_prob(coords)
        logz -= max(logz)
        z = np.exp(logz).reshape(xx.shape)
        ax.contourf(mesh_x, mesh_y, z, levels=100, cmap="viridis", antialiased=True)
        ax.contourf(mesh_x, mesh_y, z, levels=100, cmap="viridis", antialiased=True)
        ax.contourf(mesh_x, mesh_y, z, levels=100, cmap="viridis", antialiased=True)
    return ax


def save_plot(filename, plot_dir):
    if filename is None:
        filename = "particles_plot.pdf"
    if plot_dir is None:
        plot_dir = "plots"
    filename = plot_dir + "/" + filename
    plt.tight_layout()
    plt.gca().set_rasterization_zorder(0)
    plt.savefig(filename, dpi=1200)


def save_ani(ani,filename, ani_dir):
    if filename is None:
        filename = "particles_ani.gif"
    if ani_dir is None:
        ani_dir = "animations"
    filename = ani_dir + "/" + filename
    ani.save(filename, writer="pillow", dpi=150)


def plot_grid(ax, mesh_size, dims=2):

    if dims == 1:
        xlim = ax.get_xlim()
        xx = np.linspace(xlim[0], xlim[1], mesh_size)
        dx = (xlim[1]-xlim[0])/mesh_size
        return xx, dx

    if dims == 2:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        mesh_x, mesh_y = np.linspace(xlim[0], xlim[1], mesh_size), np.linspace(ylim[0], ylim[1], mesh_size)
        xx, yy = np.meshgrid(mesh_x, mesh_y)
        coords = np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)))
        return mesh_x, mesh_y, xx, yy, coords


def new_plot(compare=True, dims=2, **plt_kwargs):
    if not compare:
        fig, ax = plt.subplots(1)
        fig.clear()
        if dims == 1:
            ax = fig.add_subplot(**plt_kwargs)
        if dims == 2:
            ax = fig.add_subplot(aspect='equal', **plt_kwargs)
        return fig, ax
    if compare:
        fig, axs = plt.subplots(1, 2)
        fig.clear()
        if dims == 1:
            axs[0] = fig.add_subplot(121, **plt_kwargs)
            axs[0].set_aspect(1. / axs[0].get_data_ratio())
            #axs[0].set_yticklabels([])
            #axs[0].set_xticklabels([])
            axs[1] = fig.add_subplot(122, **plt_kwargs)
            axs[1].set_aspect(1. / axs[1].get_data_ratio())
            #axs[1].set_yticklabels([])
            #axs[1].set_xticklabels([])
        if dims == 2:
            axs[0] = fig.add_subplot(121, aspect='equal', **plt_kwargs)
            axs[0].set_yticklabels([])
            axs[0].set_xticklabels([])
            axs[1] = fig.add_subplot(122, aspect='equal', **plt_kwargs)
            axs[1].set_yticklabels([])
            axs[1].set_xticklabels([])
        fig.tight_layout()
        return fig, axs
